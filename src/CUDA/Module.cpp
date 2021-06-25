#include "CUDA/Module.h"

#include "CUDA/Utils.h"

#include "Utils/Chrono.h"

namespace CUDA {

Module::Module(void *binary, std::size_t binarySize) : m_binary(binary), m_binarySize(binarySize)
{
	auto timeLoad_start = Utils::Chrono::Start("CUDA load");

	checkDriverResult(cuModuleLoadData(&m_module, m_binary));

	Utils::Chrono::End(timeLoad_start);
}

}
