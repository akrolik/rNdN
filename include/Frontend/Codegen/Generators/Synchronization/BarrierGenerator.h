#pragma once

#include "Frontend/Codegen/Generators/Generator.h"

#include "Frontend/Codegen/Builder.h"
#include "Frontend/Codegen/Generators/Indexing/ThreadGeometryGenerator.h"

#include "PTX/Tree/Tree.h"

namespace Frontend {
namespace Codegen {

template<PTX::Bits B>
class BarrierGenerator : public Generator
{
public:
	using Generator::Generator;

	std::string Name() const override { return "BarrierGenerator"; }

	void Generate()
	{
		auto& inputOptions = this->m_builder.GetInputOptions();

		// Barrier instructions depend on the thread geometry and the number of threads still active

		if (inputOptions.IsVectorGeometry())
		{
			// All threads in the group participate in the barrier

			this->m_builder.AddStatement(new PTX::BarrierInstruction(new PTX::UInt32Value(0), true));
		}
		else if (inputOptions.IsListGeometry())
		{
			// The special barrier instruction requires threads be in multiples of the warp size

			auto warpSize = this->m_builder.GetTargetOptions().WarpSize;
			this->m_builder.SetThreadMultiple(warpSize);
			
			// Since some threads may have exited (cells out of range), count the number of active threads

			ThreadGeometryGenerator<B> geometryGenerator(this->m_builder);
			auto activeThreads = geometryGenerator.GenerateActiveThreads();
			this->m_builder.AddStatement(new PTX::BarrierInstruction(new PTX::UInt32Value(0), activeThreads, true, true));
		}
		else
		{
			Error("barrier instruction");
		}
	}
};

}
}
