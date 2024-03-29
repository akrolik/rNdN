#pragma once

#include "HorseIR/Tree/Tree.h"

namespace HorseIR {

static Module *GPUModule = new LibraryModule("GPU", {
	// Sort
	new BuiltinFunction(BuiltinFunction::Primitive::GPUOrderLib),
	new BuiltinFunction(BuiltinFunction::Primitive::GPUOrderInit),
	new BuiltinFunction(BuiltinFunction::Primitive::GPUOrder),
	new BuiltinFunction(BuiltinFunction::Primitive::GPUOrderShared),

	// Group by
	new BuiltinFunction(BuiltinFunction::Primitive::GPUGroupLib),
	new BuiltinFunction(BuiltinFunction::Primitive::GPUGroup),

	// Unique
	new BuiltinFunction(BuiltinFunction::Primitive::GPUUniqueLib),
	new BuiltinFunction(BuiltinFunction::Primitive::GPUUnique),

	// Join
	new BuiltinFunction(BuiltinFunction::Primitive::GPULoopJoinLib),
	new BuiltinFunction(BuiltinFunction::Primitive::GPULoopJoinCount),
	new BuiltinFunction(BuiltinFunction::Primitive::GPULoopJoin),

	new BuiltinFunction(BuiltinFunction::Primitive::GPUHashJoinLib),
	new BuiltinFunction(BuiltinFunction::Primitive::GPUHashJoinCreate),
	new BuiltinFunction(BuiltinFunction::Primitive::GPUHashJoinCount),
	new BuiltinFunction(BuiltinFunction::Primitive::GPUHashJoin),

	// Member
	new BuiltinFunction(BuiltinFunction::Primitive::GPUHashMemberLib),
	new BuiltinFunction(BuiltinFunction::Primitive::GPUHashMemberCreate),
	new BuiltinFunction(BuiltinFunction::Primitive::GPUHashMember),

	// Like
	new BuiltinFunction(BuiltinFunction::Primitive::GPULikeLib),
	new BuiltinFunction(BuiltinFunction::Primitive::GPULikeCacheLib),
});

}
