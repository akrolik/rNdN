#pragma once

#include <memory>
#include <string>

#include "CUDA/Device.h"

namespace CUDA {

class ExternalModule
{
public:
	ExternalModule(const std::string& name, const std::string& ptxCode) : m_name(name), m_code(ptxCode) {}

	void *GetBinary() const { return m_binary; }
	size_t GetBinarySize() const { return m_binarySize; }

	void GenerateBinary(const std::unique_ptr<Device>& device);

private:
	const std::string m_name;
	const std::string m_code;

	void *m_binary = nullptr;
	size_t m_binarySize = 0;
};

}
