#include "Runtime/Runtime.h"

namespace Runtime {

void Runtime::Initialize()
{
	// Initialize the runtime environment for the system

	m_gpu.Initialize();

	// Debug tables are used for test queries instead of real query data

	m_dataRegistry.LoadDebugData();

	// TPC-H data for benchmarking

	m_dataRegistry.LoadTPCHData();
}

}
