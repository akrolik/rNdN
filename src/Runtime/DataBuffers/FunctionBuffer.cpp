#include "Runtime/DataBuffers/FunctionBuffer.h"

#include "HorseIR/Utils/PrettyPrinter.h"

namespace Runtime {

FunctionBuffer::FunctionBuffer(const HorseIR::FunctionDeclaration *function) : DataBuffer(DataBuffer::Kind::Function), m_function(function)
{
	switch (function->GetKind())
	{
		case HorseIR::FunctionDeclaration::Kind::Definition:
		{
			m_type = new HorseIR::FunctionType(static_cast<const HorseIR::Function *>(function));
			break;
		}
		case HorseIR::FunctionDeclaration::Kind::Builtin:
		{
			m_type = new HorseIR::FunctionType(static_cast<const HorseIR::BuiltinFunction *>(function));
			break;
		}
		default:
		{
			Utils::Logger::LogError("Cannot create buffer for function '" + function->GetName() + "'");
		}
	}
	m_shape = nullptr;
}

FunctionBuffer::~FunctionBuffer()
{
	delete m_type;
	delete m_shape;
}

FunctionBuffer *FunctionBuffer::Clone() const
{
	return new FunctionBuffer(m_function);
}

std::string FunctionBuffer::Description() const
{
	return HorseIR::PrettyPrinter::PrettyString(m_function, true);
}

std::string FunctionBuffer::DebugDump(unsigned int indent, bool preindent) const
{
	std::string string;
	if (!preindent)
	{
		string += std::string(indent * Utils::Logger::IndentSize, ' ');
	}
	return string + HorseIR::PrettyPrinter::PrettyString(m_function, true);
}

}
