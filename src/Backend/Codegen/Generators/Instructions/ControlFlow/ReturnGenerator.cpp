#include "Backend/Codegen/Generators/Instructions/ControlFlow/ReturnGenerator.h"

namespace Backend {
namespace Codegen {

void ReturnGenerator::Generate(const PTX::ReturnInstruction *instruction)
{
	this->AddInstruction(new SASS::EXITInstruction());
}

}
}

