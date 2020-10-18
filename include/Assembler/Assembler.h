#pragma once

#include "Assembler/BinaryFunction.h"
#include "Assembler/BinaryProgram.h"
#include "Assembler/ELFBinary.h"

#include "SASS/SASS.h"

namespace Assembler {

class Assembler
{
public:
	ELFBinary *Assemble(const SASS::Program *program);

private:
	BinaryProgram *AssembleProgram(const SASS::Program *program);
	BinaryFunction *AssembleFunction(const SASS::Function *function);
};

}
