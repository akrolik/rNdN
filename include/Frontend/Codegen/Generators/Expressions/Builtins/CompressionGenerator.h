#pragma once

#include "HorseIR/Traversal/ConstVisitor.h"
#include "Frontend/Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "Frontend/Codegen/Builder.h"
#include "Frontend/Codegen/Generators/Expressions/MoveGenerator.h"
#include "Frontend/Codegen/Generators/Expressions/OperandCompressionGenerator.h"
#include "Frontend/Codegen/Generators/Expressions/OperandGenerator.h"
#include "Frontend/Codegen/Generators/Indexing/DataIndexGenerator.h"
#include "Frontend/Codegen/Generators/Indexing/ThreadGeometryGenerator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/Tree/Tree.h"

namespace Frontend {
namespace Codegen {

template<PTX::Bits B, class T>
class CompressionGenerator : public BuiltinGenerator<B, T>, public HorseIR::ConstVisitor
{
public:
	using BuiltinGenerator<B, T>::BuiltinGenerator;

	std::string Name() const override { return "CompressionGenerator"; }

	// The output of a compresion function handles the predicate itself. We therefore do not implement GenerateCompressionPredicate in this subclass

	PTX::Register<T> *Generate(const HorseIR::LValue *target, const std::vector<const HorseIR::Operand *>& arguments) override
	{
		// Update the resource generator with the compression information. @compress arguments:
		//   0 - bool mask
		//   1 - value

		OperandGenerator<B, PTX::PredicateType> opGen(this->m_builder);
		m_predicate = opGen.GenerateRegister(arguments.at(0), OperandGenerator<B, PTX::PredicateType>::LoadKind::Vector);

		m_target = target;
		m_arguments = arguments;
		arguments.at(1)->Accept(*this);

		return m_targetRegister;
	}

	void Visit(const HorseIR::Expression *expression) override
	{
		BuiltinGenerator<B, T>::Unimplemented("compression of non-identifier kind");
	}

	void Visit(const HorseIR::Identifier *identifier) override
	{
		auto resources = this->m_builder.GetLocalResources();

		// Compression produces a new pair (data, predicate) which is used for future
		// operation masking and writing
		// 
		// i.e. Given a compression call
		//
		//         t1:i32 = @compress(p, t0);
		//
		// the pair (t1, p) is created in the resource allocator, with data from t0

		OperandGenerator<B, T> opGen(this->m_builder);
		auto data = opGen.GenerateOperand(identifier, OperandGenerator<B, T>::LoadKind::Vector);

		// Copy the input data as it could be reassigned

		m_targetRegister = this->GenerateTargetRegister(m_target, m_arguments);

		MoveGenerator<T> moveGenerator(this->m_builder);
		moveGenerator.Generate(m_targetRegister, data);

		// Compute the predicate mask, and store it the mapping in the resources

		auto predicate = resources->template AllocateTemporary<PTX::PredicateType>();

		OperandCompressionGenerator compGen(this->m_builder);
		if (const auto compression = compGen.GetCompressionRegister(identifier))
		{
			// If a predicate has already been set on the value register, combine with the new compress predicate

			this->m_builder.AddStatement(new PTX::AndInstruction<PTX::PredicateType>(predicate, compression, m_predicate));
		}
		else
		{
			// Otherwise this is the first compression and it is simply stored

			this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::PredicateType>(predicate, m_predicate));
		}

		// Ensure that the predicate is false for out-of-bounds indexes

		DataIndexGenerator<B> indexGenerator(this->m_builder);
		auto index = indexGenerator.GenerateDataIndex();

		ThreadGeometryGenerator<B> geometryGenerator(this->m_builder);
		auto size = geometryGenerator.GenerateDataGeometry();

		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(
			predicate, nullptr, index, size, PTX::UInt32Type::ComparisonOperator::Less, predicate, PTX::PredicateModifier::BoolOperator::And
		));

		// Set the merged compression mask

		resources->SetCompressedRegister(m_targetRegister, predicate);
	}

private:
	const HorseIR::LValue *m_target = nullptr;
	std::vector<const HorseIR::Operand *> m_arguments;

	PTX::Register<T> *m_targetRegister = nullptr;
	PTX::Register<PTX::PredicateType> *m_predicate = nullptr;
};

}
}
