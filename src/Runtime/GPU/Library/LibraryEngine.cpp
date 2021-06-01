#include "Runtime/GPU/Library/LibraryEngine.h"

#include "Runtime/DataBuffers/BufferUtils.h"
#include "Runtime/DataBuffers/FunctionBuffer.h"

#include "Utils/Logger.h"

namespace Runtime {
namespace GPU {

const HorseIR::Function *LibraryEngine::GetFunction(const DataBuffer *buffer) const
{
	auto function = BufferUtils::GetBuffer<FunctionBuffer>(buffer)->GetFunction();
	if (function->GetKind() == HorseIR::FunctionDeclaration::Kind::Definition)
	{
		return static_cast<const HorseIR::Function *>(function);
	}
	Utils::Logger::LogError("GPU library cannot execute function '" + function->GetName() + "'");
}

}
}
