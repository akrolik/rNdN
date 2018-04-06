#include "CUDA/Kernel.h"

#include <iostream>

#include "CUDA/Utils.h"

namespace CUDA {

Kernel::Kernel(std::string name, std::string ptx, unsigned int parametersCount) : m_name(name), m_ptx(ptx), m_parametersCount(parametersCount)
{
	Compile();
}

void Kernel::Compile()
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

	CUlinkState linkerState;
	checkDriverResult(cuLinkCreate(optionCount, optionKeys, optionVals, &linkerState));

	CUresult result = cuLinkAddData(linkerState, CU_JIT_INPUT_PTX, (void *)m_ptx.c_str(), m_ptx.length() + 1, m_name.c_str(), 0, 0, 0);

	if (result != CUDA_SUCCESS)
	{
		std::cerr << "[Error] PTX failed to compile" << std::endl << l_errorLog << std::endl;
		std::exit(1);
	}

	checkDriverResult(cuLinkComplete(linkerState, &m_binary, &m_binarySize));
	std::cout << "[INFO] PTX compiled in " << l_wallTime << "ms" << std::endl << l_infoLog << std::endl;

	checkDriverResult(cuLinkDestroy(linkerState));
}

}
