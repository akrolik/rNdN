#pragma once

#include <string>

#include "SASS/Node.h"

#include "SASS/Instructions/Instruction.h"

namespace SASS {

class Relocation : public Node
{
public:
	enum class Kind {
		ABS24_20,
		ABS32_LO_20,
		ABS32_HI_20
	};

	static std::string KindString(Kind kind)
	{
		switch (kind)
		{
			case Kind::ABS24_20:
				return "ABS24_20";
			case Kind::ABS32_LO_20:
				return "ABS32_LO_20";
			case Kind::ABS32_HI_20:
				return "ABS32_HI_20";
		}
		return "<unknown>";
	}

	Relocation(const Instruction *instruction, const std::string& name, Kind kind) : m_instruction(instruction), m_name(name), m_kind(kind) {}
	
	// Properties

	const Instruction *GetInstruction() const { return m_instruction; }
	void SetInstruction(const Instruction *instruction) { m_instruction = instruction; }

	const std::string& GetName() const { return m_name; }
	void SetName(const std::string& name) { m_name = name; }

	Kind GetKind() const { return m_kind; }
	void SetKind(Kind kind) { m_kind = kind; }

	// Formatting

	std::string ToString() const override
	{
		return ".reloc " + m_name + " " + KindString(m_kind) + " (" + m_instruction->ToString() + ")";
	}

private:
	const Instruction *m_instruction = nullptr;
	std::string m_name;
	Kind m_kind;
};

}
