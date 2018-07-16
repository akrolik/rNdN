#include "CUDA/libdevice.h"

#include <iostream>
#include <chrono>

#include "llvm/IRReader/IRReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/CodeGen/CommandFlags.def"
#include "llvm/IR/Module.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

namespace CUDA {

std::string Generate(const std::string& compute)
{
	auto timeDummy_start = std::chrono::steady_clock::now();

	// Initialize dummy LLVM program with definitions of all required math functions.
	// This will allow us to extract only the needed functions without writing a
	// traversal pass

	std::string dummy = "";

	std::vector<std::string> unaryFunctions({"cos", "sin", "tan", "acos", "asin", "atan", "cosh", "sinh", "tanh", "acosh", "asinh", "atanh", "log", "exp"});
	for (const auto& function : unaryFunctions)
	{
		dummy += "declare float @__nv_" + function + "f(float)\n";
		dummy += "declare double @__nv_" + function + "(double)\n";
	}

	std::vector<std::string> binaryFunctions({"pow", "modf"});
	for (const auto& function : binaryFunctions)
	{
		dummy += "declare float @__nv_" + function + "f(float,float)\n";
		dummy += "declare double @__nv_" + function + "(double,double)\n";
	}

	auto dummyBuffer = llvm::MemoryBuffer::getMemBuffer(llvm::StringRef(dummy));

	// Parse the LLVM IR into a module

	llvm::SMDiagnostic diagnostic;
	llvm::LLVMContext context;
	auto module = llvm::parseIR(*dummyBuffer, diagnostic, context);
	if (module == nullptr)
	{
		std::cerr << "[LLVM Error] Cannot parse dummy definitions : " << diagnostic.getMessage().str() << std::endl;
		std::exit(EXIT_FAILURE);
	}

	// Initialize the module target and data layout. These are required for the
	// code generation pass and are given by the NVidia code

	std::string targetString = "nvptx64-nvidia-cuda";
	std::string dataLayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64";

	module->setTargetTriple(targetString);
	module->setDataLayout(dataLayout);

	auto timeDummy_end = std::chrono::steady_clock::now();

	// Load the libdevice math library from the default CUDA path and parse
	// into an LLVM module

	auto timeLib_start = std::chrono::steady_clock::now();

	std::string libdevicePath = "/usr/local/cuda/nvvm/libdevice/libdevice.10.bc";

	auto libModule = llvm::parseIRFile(libdevicePath, diagnostic, context);
	if (libModule == nullptr)
	{
		std::cerr << "[LLVM Error] Cannot parse libdevice file " << diagnostic.getFilename().str() << " : " << diagnostic.getMessage().str() << std::endl;
		std::exit(EXIT_FAILURE);
	}

	libModule->setTargetTriple(targetString);
	libModule->setDataLayout(dataLayout);

	auto timeLib_end = std::chrono::steady_clock::now();

	// Link the libdevice module into the dummy module, keeping only methods that
	// are referenced in the dummy

	auto timeLink_start = std::chrono::steady_clock::now();

	llvm::Linker linker(*module);
	if (linker.linkInModule(std::move(libModule), llvm::Linker::Flags::LinkOnlyNeeded))
	{
		std::cerr << "[LLVM Error] Error linking dummy and libdevice" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	auto timeLink_end = std::chrono::steady_clock::now();

	// Initialize the LLVM NVPTX target for code generation

	auto timeOptimize_start = std::chrono::steady_clock::now();

	LLVMInitializeNVPTXTarget();
	LLVMInitializeNVPTXTargetInfo();
	LLVMInitializeNVPTXTargetMC();
	LLVMInitializeNVPTXAsmPrinter();

	// Initialize optimization passes required to generate efficient code for our target

	llvm::legacy::FunctionPassManager functionPasses(module.get());
	llvm::legacy::PassManager modulePasses;

	// Optimize for speed and not size

	llvm::PassManagerBuilder builder;
	builder.OptLevel = 3;
	builder.SizeLevel = 0;
	builder.LoopVectorize = true;
	builder.SLPVectorize = true;

	// Create the target machine settings. These are used to configure the optimization
	// passes as well as creating the code generation pass

	std::string error;
	const llvm::Target *target = llvm::TargetRegistry::lookupTarget(targetString, error);
	if (target == nullptr)
	{
		std::cerr << "[LLVM Error] Cannot lookup target '" << targetString << "' : " << error << std::endl;
		std::exit(EXIT_FAILURE);
	}

	llvm::TargetOptions targetOptions = InitTargetOptionsFromCodeGenFlags();
	targetOptions.AllowFPOpFusion = llvm::FPOpFusion::Fast;
	targetOptions.UnsafeFPMath = true;
	targetOptions.NoInfsFPMath = true;
	targetOptions.NoNaNsFPMath = true;

	//TODO: Look into getting the +ptx value from llvm
	auto machine(target->createTargetMachine(
		targetString,
		compute.c_str(),
		"+ptx50",
		targetOptions,
		llvm::Optional<llvm::Reloc::Model>(),
		llvm::Optional<llvm::CodeModel::Model>(),
		llvm::CodeGenOpt::Aggressive
	));

	machine->adjustPassManager(builder);

	// Generate the passes

	builder.populateFunctionPassManager(functionPasses);
	builder.populateModulePassManager(modulePasses);

	// Execute the optimization pipeline

	functionPasses.doInitialization();
	for (auto it = module->begin(); it != module->end(); ++it)
	{
		functionPasses.run(*it);
	}
	functionPasses.doFinalization();
	modulePasses.run(*module);

	auto timeOptimize_end = std::chrono::steady_clock::now();

	// Generate PTX code from the optimized LLVM module. We generate to a string
	// instead of a file so it links with the next phase of the pipeline

	auto timeCode_start = std::chrono::steady_clock::now();

	llvm::legacy::PassManager codegenPasses;
	std::string ptxCode;
	llvm::raw_string_ostream stream(ptxCode);
	llvm::buffer_ostream pstream(stream);
	machine->addPassesToEmitFile(codegenPasses, pstream, llvm::TargetMachine::CGFT_AssemblyFile, true);
	codegenPasses.run(*module);

	auto timeCode_end = std::chrono::steady_clock::now();

	// Output timing info for diagnostics

	auto dummyTime = std::chrono::duration_cast<std::chrono::microseconds>(timeDummy_end - timeDummy_start).count();
	auto libdeviceTime = std::chrono::duration_cast<std::chrono::microseconds>(timeLib_end - timeLib_start).count();
	auto optimizeTime = std::chrono::duration_cast<std::chrono::microseconds>(timeOptimize_end - timeOptimize_start).count();
	auto codegenTime = std::chrono::duration_cast<std::chrono::microseconds>(timeCode_end - timeCode_start).count();

	std::cout << "[INFO] libdevice PTX generated in " << dummyTime + libdeviceTime + optimizeTime + codegenTime << " mus" << std::endl;
	std::cout << "         - dummy load: " << dummyTime << " mus" << std::endl;
	std::cout << "         - libdevice load: " << libdeviceTime << " mus" << std::endl;
	std::cout << "         - optimize: " << optimizeTime << " mus" << std::endl;
	std::cout << "         - codegen: " << codegenTime << " mus" << std::endl;

	return ptxCode;
}

ExternalModule libdevice::CreateModule(const Device& device)
{
	ExternalModule module("libdevice", Generate(device.GetComputeCapability()));
	module.GenerateBinary(device);
	return module;
}

}
