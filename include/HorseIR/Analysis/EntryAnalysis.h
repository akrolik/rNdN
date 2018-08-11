#pragma once

#include "HorseIR/Traversal/ForwardTraversal.h"

#include "HorseIR/Tree/Method.h"
#include "HorseIR/Tree/Program.h"

namespace HorseIR {

class EntryAnalysis : public ForwardTraversal
{
public:
	void Analyze(Program *program);
	Method *GetEntry() const { return m_entry; }

	void Visit(Method *method) override;

private:
	Method *m_entry = nullptr;
};

}
