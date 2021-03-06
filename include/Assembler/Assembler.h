#pragma once

#include "Assembler/BinaryFunction.h"
#include "Assembler/BinaryProgram.h"

#include "SASS/SASS.h"

namespace Assembler {

class Assembler
{
public:
	BinaryProgram *Assemble(const SASS::Program *program);
	BinaryFunction *Assemble(const SASS::Function *function);
};

}
