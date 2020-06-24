#include "Runtime/Runtime.h"

#include "CUDA/Buffer.h"
#include "CUDA/BufferManager.h"
#include "CUDA/Vector.h"

#include "Utils/Chrono.h"
#include "Utils/Options.h"

namespace Runtime {

Runtime *Runtime::s_instance = nullptr;

void Runtime::Initialize()
{
	// Initialize the runtime environment for the system

	m_gpu.Initialize();

	CUDA::BufferManager::Initialize();
}

void Runtime::LoadData()
{
	auto timeData_start = Utils::Chrono::Start("Load data");

	// Debug tables are used for test queries instead of real query data

	m_dataRegistry.LoadDebugData();

	if (Utils::Options::Present(Utils::Options::Opt_Load_tpch))
	{
		// TPC-H data for benchmarking

		m_dataRegistry.LoadTPCHData();
	}

	auto timeDummy_start = Utils::Chrono::Start("CUDA dummy initialization");

	// Transfer dummy data to initialize the PCI-e bus

	CUDA::Vector<std::uint64_t> dummyVector;
	dummyVector.resize(1);
	auto dummyBuffer = new CUDA::Buffer(dummyVector.size() * sizeof(std::uint64_t));
	dummyBuffer->SetTag("dummy");
	dummyBuffer->SetCPUBuffer(dummyVector.data());
	dummyBuffer->AllocateOnGPU();
	dummyBuffer->TransferToGPU();
	delete dummyBuffer;

	Utils::Chrono::End(timeDummy_start);
	Utils::Chrono::End(timeData_start);
}

Runtime::~Runtime()
{
	CUDA::BufferManager::Destroy();
}

}
