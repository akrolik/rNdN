#pragma once

#include "SASS/Node.h"

#include "Utils/Format.h"

#include <string>

namespace SASS {

class Variable : public Node
{
public:
	Variable(const std::string& name, std::size_t size, std::size_t dataSize)
		: m_name(name), m_size(size), m_dataSize(dataSize) {}
	
	// Properties

	const std::string& GetName() const { return m_name; }
	void SetName(const std::string& name) { m_name = name; }

	std::size_t GetSize() const { return m_size; }
	void SetSize(std::size_t size) { m_size = size; }

	std::size_t GetDataSize() const { return m_dataSize; }
	void SetDataSize(std::size_t dataSize) { m_dataSize = dataSize; }

	// Formatting

	std::string ToString() const override
	{
		auto string = "." + SpaceName() + " " + m_name + " { ";
		string += "size = " + Utils::Format::HexString(m_size) + " bytes; ";
		string += "datasize = " + Utils::Format::HexString(m_dataSize) + " bytes }";
		return string;
	}

private:
	virtual std::string SpaceName() const = 0 ;

	std::string m_name;
	std::size_t m_size = 0;
	std::size_t m_dataSize = 0;
};

}
