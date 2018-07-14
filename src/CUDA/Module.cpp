#include "CUDA/Module.h"

#include "CUDA/Utils.h"

#include <iostream>
#include <chrono>

#include "llvm/IRReader/IRReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
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

Module::Module(std::string ptx) : m_ptx(ptx)
{
	Compile();
}

std::string Compile_libdevice()
{
	auto time0 = std::chrono::steady_clock::now();

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

	llvm::SMDiagnostic diagnostic;
	llvm::LLVMContext context;
	auto module = llvm::parseIR(*dummyBuffer, diagnostic, context);
	if (module == nullptr)
	{
		std::cerr << "[LLVM Error] Cannot parse libdevice file " << diagnostic.getFilename().str() << " : " << diagnostic.getMessage().str() << std::endl;
		std::exit(EXIT_FAILURE);
	}

	std::string targetString = "nvptx64-nvidia-cuda";
	std::string dataLayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64";

	module->setTargetTriple(targetString);
	module->setDataLayout(dataLayout);

	auto time1 = std::chrono::steady_clock::now();

	std::string libdevicePath = "/usr/local/cuda/nvvm/libdevice/libdevice.10.bc";

	auto libModule = llvm::parseIRFile(libdevicePath, diagnostic, context);
	if (libModule == nullptr)
	{
		std::cerr << "[LLVM Error] Cannot parse libdevice file " << diagnostic.getFilename().str() << " : " << diagnostic.getMessage().str() << std::endl;
		std::exit(EXIT_FAILURE);
	}

	libModule->setTargetTriple(targetString);
	libModule->setDataLayout(dataLayout);

	auto time2 = std::chrono::steady_clock::now();

	llvm::Linker linker(*module);
	if (linker.linkInModule(std::move(libModule), llvm::Linker::Flags::LinkOnlyNeeded))
	{
		std::cerr << "[LLVM Error] Error linking dummy and libdevice" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	auto time3 = std::chrono::steady_clock::now();

	LLVMInitializeNVPTXTarget();
	LLVMInitializeNVPTXTargetInfo();
	LLVMInitializeNVPTXTargetMC();
	LLVMInitializeNVPTXAsmPrinter();

	llvm::legacy::FunctionPassManager functionPasses(module.get());
	llvm::legacy::PassManager modulePasses;

	llvm::PassManagerBuilder builder;
	builder.OptLevel = 3;
	builder.SizeLevel = 0;
	builder.LoopVectorize = true;
	builder.SLPVectorize = true;

	std::string error;
	const llvm::Target *target = llvm::TargetRegistry::lookupTarget(targetString, error);
	if (target == nullptr)
	{
		std::cerr << "[LLVM Error] Cannot lookup target '" << targetString << "' : " << error << std::endl;
		std::exit(EXIT_FAILURE);
	}

	//TODO: Verify all these options
	llvm::TargetOptions targetOptions;
	targetOptions.PrintMachineCode = false;
	targetOptions.AllowFPOpFusion = llvm::FPOpFusion::Fast;
	targetOptions.UnsafeFPMath = true;
	targetOptions.NoInfsFPMath = true;
	targetOptions.NoNaNsFPMath = true;
	targetOptions.HonorSignDependentRoundingFPMathOption = false;
	targetOptions.NoZerosInBSS = false;

	//TODO: Use sm_ value from the device
	//TODO: Look into getting the +ptx value from llvm
	auto machine(target->createTargetMachine(
		targetString,
		"sm_61",
		"+ptx50",
		targetOptions,
		llvm::Optional<llvm::Reloc::Model>(),
		llvm::Optional<llvm::CodeModel::Model>(),
		llvm::CodeGenOpt::Aggressive
	));

	machine->adjustPassManager(builder);

	builder.populateFunctionPassManager(functionPasses);
	builder.populateModulePassManager(modulePasses);

	functionPasses.doInitialization();
	for (auto it = module->begin(); it != module->end(); ++it)
	{
		functionPasses.run(*it);
	}
	functionPasses.doFinalization();
	modulePasses.run(*module);

	auto time4 = std::chrono::steady_clock::now();

	llvm::legacy::PassManager codegenPasses;
	std::string ptxcode;
	llvm::raw_string_ostream stream(ptxcode);
	llvm::buffer_ostream pstream(stream);
	machine->addPassesToEmitFile(codegenPasses, pstream, llvm::TargetMachine::CGFT_AssemblyFile, true);
	codegenPasses.run(*module);

	auto time5 = std::chrono::steady_clock::now();

	auto dummyTime = std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1).count();
	auto libdeviceTime = std::chrono::duration_cast<std::chrono::microseconds>(time3 - time2).count();
	auto optimizeTime = std::chrono::duration_cast<std::chrono::microseconds>(time4 - time3).count();
	auto codegenTime = std::chrono::duration_cast<std::chrono::microseconds>(time5 - time4).count();

	std::cout << "[INFO] libdevice PTX generated in " << dummyTime + libdeviceTime + optimizeTime + codegenTime << " mus" << std::endl;
	std::cout << "         - dummy load: " << dummyTime << " mus" << std::endl;
	std::cout << "         - libdevice load: " << libdeviceTime << " mus" << std::endl;
	std::cout << "         - optimize: " << optimizeTime << " mus" << std::endl;
	std::cout << "         - codegen: " << codegenTime << " mus" << std::endl;

	return ptxcode;
}

void Module::Compile()
{
	CUjit_option optionKeys[6];
	void *optionVals[6];
	unsigned int optionCount = 6;

	float l_wallTime;
	char l_infoLog[8192];
	char l_errorLog[8192];
	unsigned int l_logSize = 8192;

	optionKeys[0] = CU_JIT_WALL_TIME;
	optionVals[0] = (void *)&l_wallTime;

	optionKeys[1] = CU_JIT_INFO_LOG_BUFFER;
	optionVals[1] = (void *)l_infoLog;

	optionKeys[2] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
	optionVals[2] = (void *)(long)l_logSize;

	optionKeys[3] = CU_JIT_ERROR_LOG_BUFFER;
	optionVals[3] = (void *)l_errorLog;

	optionKeys[4] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
	optionVals[4] = (void *)(long)l_logSize;

	optionKeys[5] = CU_JIT_LOG_VERBOSE;
	optionVals[5] = (void *)true;

	//TODO: Investigate additional JIT options
	// CU_JIT_FAST_COMPILE
	// CU_JIT_CACHE_MODE
	// CU_JIT_GENERATE_LINE_INFO

	CUlinkState linkerState;
	checkDriverResult(cuLinkCreate(optionCount, optionKeys, optionVals, &linkerState));

	auto time0 = std::chrono::steady_clock::now();
	std::string libdevice_ptx = Compile_libdevice();
	auto time1 = std::chrono::steady_clock::now();

	CUresult result;
	result = cuLinkAddData(linkerState, CU_JIT_INPUT_PTX, (void *)libdevice_ptx.c_str(), libdevice_ptx.length() + 1, nullptr, 0, nullptr, nullptr);
	if (result != CUDA_SUCCESS)
	{
		std::cerr << "[ERROR] libdevice PTX failed to compile" << std::endl << l_errorLog << std::endl;
		std::exit(EXIT_FAILURE);
	}
	auto time2 = std::chrono::steady_clock::now();

	result = cuLinkAddData(linkerState, CU_JIT_INPUT_PTX, (void *)m_ptx.c_str(), m_ptx.length() + 1, nullptr, 0, nullptr, nullptr);
	if (result != CUDA_SUCCESS)
	{
		std::cerr << "[ERROR] PTX failed to compile" << std::endl << l_errorLog << std::endl;
		std::exit(EXIT_FAILURE);
	}

	auto time3 = std::chrono::steady_clock::now();
	checkDriverResult(cuLinkComplete(linkerState, &m_binary, &m_binarySize));
	auto time4 = std::chrono::steady_clock::now();

	auto libdeviceTime = std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1).count();
	auto kernelTime = std::chrono::duration_cast<std::chrono::microseconds>(time3 - time2).count();
	auto linkTime = std::chrono::duration_cast<std::chrono::microseconds>(time4 - time3).count();

	std::cout << "[INFO] PTX compiled in " << libdeviceTime + kernelTime + linkTime << " mus" << std::endl;
	std::cout << "         - libdevice: " << libdeviceTime << " mus" << std::endl;
	std::cout << "         - kernel: " << kernelTime << " mus" << std::endl;
	std::cout << "         - link: " << linkTime << " mus" << std::endl;
	std::cout << l_infoLog << std::endl;

 	checkDriverResult(cuModuleLoadData(&m_module, m_binary));
	checkDriverResult(cuLinkDestroy(linkerState));
}

}
