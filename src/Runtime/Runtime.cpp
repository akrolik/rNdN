#include "Runtime/Runtime.h"

#include "Utils/Options.h"

namespace Runtime {

void Runtime::Initialize()
{
	// Initialize the runtime environment for the system

	m_gpu.Initialize();

	// Debug tables are used for test queries instead of real query data

	m_dataRegistry.LoadDebugData();

	if (Utils::Options::Present(Utils::Options::Opt_Load_tpch))
	{
		// TPC-H data for benchmarking

		m_dataRegistry.LoadTPCHData();
	}
}

}
