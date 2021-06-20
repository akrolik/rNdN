#pragma once

#include "HorseIR/Tree/Node.h"

namespace HorseIR {

class Statement : public Node
{
public:
	Statement(int line = 0) : m_line(line) {}

	// Line number

	int GetLineNumber() const { return m_line; }
	void SetLineNumber(int line) { m_line = line; }

	virtual Statement *Clone() const override = 0;

private:
	int m_line = 0;
};

}
