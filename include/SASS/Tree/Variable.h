#pragma once

#include "SASS/Tree/Node.h"

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

private:
	std::string m_name;
	std::size_t m_size = 0;
	std::size_t m_dataSize = 0;
};

}
