#pragma once

#include <cstddef>

namespace Assembler {

class ELFBinary
{
public:
	ELFBinary(void *data, std::size_t size) : m_data(data), m_size(size) {}

	void *GetData() const { return m_data; }
	std::size_t GetSize() const { return m_size; }

private:
	void *m_data = nullptr;
	std::size_t m_size = 0;
};

}
