#pragma once

#include "Backend/Codegen/Builder.h"

#include "PTX/Utils/PrettyPrinter.h"

#include "Utils/Logger.h"

#include <string>

namespace Backend { 
namespace Codegen {

class ArchitectureDispatch;
class Generator
{
public:
	friend class ArchitectureDispatch;

	Generator(Builder& builder) : m_builder(builder) {}

	[[noreturn]] void Error(const std::string& message)
	{
		Utils::Logger::LogError(Name() + ": Unable to generate " + message);
	}

	[[noreturn]] void Error(const PTX::Node *node, const std::string& message)
	{
		Error("'" + PTX::PrettyPrinter::PrettyString(node) + "', " + message);
	}

	[[noreturn]] void Error(const PTX::Node *node)
	{
		Error("'" + PTX::PrettyPrinter::PrettyString(node) + "'");
	}

	virtual std::string Name() const = 0;

protected:
	Builder &m_builder;
};

}
}
