#pragma once

#include "SASS/Node.h"

#include "Utils/Format.h"

#include <string>

namespace SASS {

class GlobalVariable : public Node
{
public:
	GlobalVariable(const std::string& name, std::size_t offset, std::size_t size) : m_name(name), m_offset(offset), m_size(size) {}
	
	// Properties

	const std::string& GetName() const { return m_name; }
	void SetName(const std::string& name) { m_name = name; }

	std::size_t GetOffset() const { return m_offset; }
	void SetOffset(std::size_t offset) { m_offset = offset; }

	std::size_t GetSize() const { return m_size; }
	void SetSize(std::size_t size) { m_size = size; }

	// Formatting

	std::string ToString() const override
	{
		return ".global " + m_name + " (offset=" + Utils::Format::HexString(m_offset) + ", size=" + Utils::Format::HexString(m_size) + ")";
	}

private:
	std::string m_name;
	std::size_t m_offset = 0;
	std::size_t m_size = 0;
};

}
