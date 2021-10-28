#include "CUDA/libdevice.h"

#include "llvm/IRReader/IRReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"

namespace CUDA {

std::string Generate(const std::string& compute)
{
	auto timeLibrary_start = Utils::Chrono::Start("Generate external library 'libdevice'");
	auto timeDummy_start = Utils::Chrono::Start("Dummy LLVM library");

	// Initialize dummy LLVM program with definitions of all required math functions.
	// This will allow us to extract only the needed functions without writing a
	// traversal pass

	std::string dummy = "";

	std::vector<std::string> unaryFunctions({"cos", "sin", "tan", "acos", "asin", "atan", "cosh", "sinh", "tanh", "acosh", "asinh", "atanh", "log", "log2", "log10", "sqrt", "exp"});
	for (const auto& function : unaryFunctions)
	{
		dummy += "declare float @__nv_" + function + "f(float)\n";
		dummy += "declare double @__nv_" + function + "(double)\n";
	}

	std::vector<std::string> binaryFunctions({"pow", "fmod"});
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
		Utils::Logger::LogError("Cannot parse dummy definitions : " + diagnostic.getMessage().str(), "LLVM Error");
	}

	// Initialize the module target and data layout. These are required for the
	// code generation pass and are given by the NVidia code

	std::string targetString = "nvptx64-nvidia-cuda";
	std::string dataLayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64";

	module->setTargetTriple(targetString);
	module->setDataLayout(dataLayout);

	Utils::Chrono::End(timeDummy_start);

	// Load the libdevice math library from the default CUDA path and parse
	// into an LLVM module

	auto timeParse_start = Utils::Chrono::Start("Parse LLVM");

	std::string libdevicePath = "/usr/local/cuda/nvvm/libdevice/libdevice.10.bc";

	auto libModule = llvm::parseIRFile(libdevicePath, diagnostic, context);
	if (libModule == nullptr)
	{
		Utils::Logger::LogError("Cannot parse libdevice file " + diagnostic.getFilename().str() + " : " + diagnostic.getMessage().str(), "LLVM Error");
	}

	libModule->setTargetTriple(targetString);
	libModule->setDataLayout(dataLayout);

	Utils::Chrono::End(timeParse_start);

	// Link the libdevice module into the dummy module, keeping only functions that
	// are referenced in the dummy

	auto timeLink_start = Utils::Chrono::Start("Link");

	llvm::Linker linker(*module);
	if (linker.linkInModule(std::move(libModule), llvm::Linker::Flags::LinkOnlyNeeded))
	{
		Utils::Logger::LogError("Error linking dummy and libdevice", "LLVM Error");
	}

	Utils::Chrono::End(timeLink_start);

	// Initialize the LLVM NVPTX target for code generation

	auto timeOptimize_start = Utils::Chrono::Start("Optimize"); 
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
		Utils::Logger::LogError("Cannot lookup target '" + targetString + "' : " + error, "LLVM ERROR");
	}

	llvm::TargetOptions targetOptions;
	targetOptions.AllowFPOpFusion = llvm::FPOpFusion::Fast;
	targetOptions.UnsafeFPMath = true;
	targetOptions.NoInfsFPMath = true;
	targetOptions.NoNaNsFPMath = true;
	targetOptions.MCOptions = llvm::mc::InitMCTargetOptionsFromFlags();

	auto machine(target->createTargetMachine(
		targetString,
		compute.c_str(),
		compute.c_str(), // Allow LLVM to choose the PTX version
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

	Utils::Chrono::End(timeOptimize_start);

	// Generate PTX code from the optimized LLVM module. We generate to a string
	// instead of a file so it links with the next phase of the pipeline

	auto timeEmit_start = Utils::Chrono::Start("Emit");

	// Force flush on scope exit

	std::string ptxCode;
	{
		llvm::legacy::PassManager codegenPasses;
		llvm::raw_string_ostream stream(ptxCode);
		llvm::buffer_ostream pstream(stream);
		machine->addPassesToEmitFile(codegenPasses, pstream, nullptr, llvm::CGFT_AssemblyFile, true);
		codegenPasses.run(*module);
	}

	Utils::Chrono::End(timeEmit_start);
	Utils::Chrono::End(timeLibrary_start);

	return ptxCode;
}

ExternalModule libdevice::CreateModule(const std::unique_ptr<Device>& device)
{
	ExternalModule module("libdevice", Generate(device->GetComputeCapability()));
	module.GenerateBinary(device);
	return module;
}

}
