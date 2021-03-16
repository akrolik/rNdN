#pragma once

#include "Frontend/Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "Frontend/Codegen/Builder.h"
#include "Frontend/Codegen/Generators/Data/TargetGenerator.h"
#include "Frontend/Codegen/Generators/Expressions/OperandGenerator.h"
#include "Frontend/Codegen/Generators/Expressions/Builtins/ComparisonGenerator.h"
#include "Frontend/Codegen/Generators/Indexing/DataIndexGenerator.h"
#include "Frontend/Codegen/Generators/Indexing/DataSizeGenerator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/Tree/Tree.h"

namespace Frontend {
namespace Codegen {

template<PTX::Bits B, class T>
class UniqueGenerator : public BuiltinGenerator<B, T>
{
public:
	using BuiltinGenerator<B, T>::BuiltinGenerator;

	std::string Name() const override { return "UniqueGenerator"; }
};

template<PTX::Bits B>
class UniqueGenerator<B, PTX::Int64Type>: public BuiltinGenerator<B, PTX::Int64Type>
{
public:
	using BuiltinGenerator<B, PTX::Int64Type>::BuiltinGenerator;

	std::string Name() const override { return "UniqueGenerator"; }

	PTX::Register<PTX::Int64Type> *Generate(const HorseIR::LValue *target, const std::vector<const HorseIR::Operand *>& arguments) override
	{
		DispatchType(*this, arguments.at(0)->GetType(), target, arguments);

		return m_targetRegister;
	}

	template<class T>
	void GenerateVector(const HorseIR::LValue *target, const std::vector<const HorseIR::Operand *>& arguments)
	{
		auto resources = this->m_builder.GetLocalResources();
		auto dataArgument = arguments.at(0);

		// Get the current data item

		DataIndexGenerator<B> indexGenerator(this->m_builder);
		auto index = indexGenerator.GenerateDataIndex();

		DataSizeGenerator<B> sizeGenerator(this->m_builder);
		auto size = sizeGenerator.GenerateSize(dataArgument);

		// Store the index as output of the unique function

		m_targetRegister = this->GenerateTargetRegister(target, arguments);
		ConversionGenerator::ConvertSource<PTX::Int64Type, PTX::UInt32Type>(this->m_builder, m_targetRegister, index);

		// Compare against all higher indexes

		auto uniquePredicate = resources->template AllocateTemporary<PTX::PredicateType>();
		auto loopIndex = resources->template AllocateTemporary<PTX::UInt32Type>();
		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::PredicateType>(uniquePredicate, new PTX::BoolValue(true)));
		this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(loopIndex, index, new PTX::UInt32Value(1)));

		this->m_builder.AddIfStatement("UNIQUE_SKIP", [&]()
		{
			auto initPredicate = resources->template AllocateTemporary<PTX::PredicateType>();
			this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(
				initPredicate, loopIndex, size, PTX::UInt32Type::ComparisonOperator::GreaterEqual
			));
			return std::make_tuple(initPredicate, false);
		},
		[&]()
		{
			OperandGenerator<B, T> operandGenerator(this->m_builder);
			operandGenerator.SetBoundsCheck(false);
			auto data = operandGenerator.GenerateOperand(dataArgument, OperandGenerator<B, T>::LoadKind::Vector);

			this->m_builder.AddDoWhileLoop("UNIQUE", [&](Builder::LoopContext& loopContext)
			{
				auto nextData = operandGenerator.GenerateOperand(dataArgument, loopIndex, this->m_builder.UniqueIdentifier("unique"));

				auto comparisonPredicate = resources->template AllocateTemporary<PTX::PredicateType>();
				ComparisonGenerator<B, PTX::PredicateType> comparisonGenerator(this->m_builder, ComparisonOperation::NotEqual);
				comparisonGenerator.Generate(comparisonPredicate, data, nextData);

				this->m_builder.AddStatement(new PTX::AndInstruction<PTX::PredicateType>(uniquePredicate, uniquePredicate, comparisonPredicate));
				this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(loopIndex, loopIndex, new PTX::UInt32Value(1)));

				// Loop predicate

				auto endPredicate = resources->template AllocateTemporary<PTX::PredicateType>();
				this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(
					endPredicate, loopIndex, size, PTX::UInt32Type::ComparisonOperator::Less
				));
				return std::make_tuple(endPredicate, false);
			});
		});

		// Predicate the output using the unique predicate

		resources->SetCompressedRegister(m_targetRegister, uniquePredicate);
	}

	template<class T>
	void GenerateList(const HorseIR::LValue *target, const std::vector<const HorseIR::Operand *>& arguments)
	{
		if (this->m_builder.GetInputOptions().IsVectorGeometry())
		{
			BuiltinGenerator<B, PTX::Int64Type>::Unimplemented("list-in-vector");
		}

		// Lists are handled by the vector code through a projection

		GenerateVector<T>(target, arguments);
	}

	template<class T>
	void GenerateTuple(unsigned int index, const HorseIR::LValue *target, const std::vector<const HorseIR::Operand *>& arguments)
	{
		BuiltinGenerator<B, PTX::Int64Type>::Unimplemented("list-in-vector");
	}

private:
	PTX::Register<PTX::Int64Type> *m_targetRegister = nullptr;
};

}
}
