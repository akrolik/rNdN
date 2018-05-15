#pragma once

namespace PTX {

class PredicatedInstruction : public InstructionStatement
{
public:
	void SetPredicate(Register<PredicateType> *predicate, bool negate = false) { m_predicate = predicate; m_negatePredicate = negate; }

	std::string ToString() const
	{
		std::ostringstream code;
		if (m_predicate != nullptr)
		{
			code << "@";
			if (m_negatePredicate)
			{
				code << "!";
			}
			code << m_predicate->GetName() << " ";
		}
		code << InstructionStatement::ToString();
		return code.str();
	}

private:
	Register<PredicateType> *m_predicate = nullptr;
	bool m_negatePredicate = false;
};

}
