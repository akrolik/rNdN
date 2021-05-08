#pragma once

#include <string>

#include "SASS/Tree/Node.h"

#include "SASS/Tree/Instructions/Instruction.h"

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

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }
	
private:
	const Instruction *m_instruction = nullptr;
	std::string m_name;
	Kind m_kind;
};

}
