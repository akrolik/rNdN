#include "SASS/Traversal/Visitor.h"

#include "SASS/SASS.h"

namespace SASS {

void Visitor::Visit(SSYInstruction *instruction)
{
	Visit(static_cast<DivergenceInstruction *>(instruction));
}

void Visitor::Visit(PBKInstruction *instruction)
{
	Visit(static_cast<DivergenceInstruction *>(instruction));
}

void Visitor::Visit(PCNTInstruction *instruction)
{
	Visit(static_cast<DivergenceInstruction *>(instruction));
}

}
