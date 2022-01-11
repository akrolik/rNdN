#pragma once

#include <tuple>

#include "Frontend/Codegen/Generators/Generator.h"
#include "HorseIR/Traversal/ConstVisitor.h"

#include "Frontend/Codegen/Builder.h"
#include "Frontend/Codegen/Generators/Expressions/OperandGenerator.h"
#include "Frontend/Codegen/Generators/Expressions/Builtins/ComparisonGenerator.h"

#include "HorseIR/Analysis/Shape/Shape.h"
#include "HorseIR/Analysis/Shape/ShapeUtils.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/Tree/Tree.h"

namespace Frontend {
namespace Codegen {

template<PTX::Bits B>
class InternalHashGenerator : public Generator, public HorseIR::ConstVisitor
{
public:
	using Generator::Generator;

	std::string Name() const override { return "InternalHashGenerator"; }

	PTX::Register<PTX::UInt32Type> *Generate(const HorseIR::Operand *operand, const std::vector<ComparisonOperation>& comparisonOperations)
	{
		m_hash = nullptr;
		m_comparisonOperations = comparisonOperations;
		operand->Accept(*this);
		return m_hash;
	}

	void Visit(const HorseIR::Identifier *identifier) override
	{
		DispatchType(*this, identifier->GetType(), identifier);
	}

	void Visit(const HorseIR::Literal *literal) override
	{
		Error("literal data for @hash_create");
	}

	template<class T>
	void GenerateVector(const HorseIR::Identifier *identifier)
	{
		m_hash = GenerateHash<T>(identifier);
	}

	template<class T>
	void GenerateList(const HorseIR::Identifier *identifier)
	{
		const auto& inputOptions = this->m_builder.GetInputOptions();
		const auto parameter = inputOptions.Parameters.at(identifier->GetSymbol());
		const auto shape = inputOptions.ParameterShapes.at(parameter);

		if (const auto listShape = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::ListShape>(shape))
		{
			if (const auto size = HorseIR::Analysis::ShapeUtils::GetSize<HorseIR::Analysis::Shape::ConstantSize>(listShape->GetListSize()))
			{
				for (auto index = 0u; index < size->GetValue(); ++index)
				{
					GenerateTuple<T>(index, identifier);
				}
				return;
			}
		}
		Error("non-constant cell count");
	}

	template<class T>
	void GenerateTuple(unsigned int index, const HorseIR::Identifier *identifier)
	{
		if (auto hash = GenerateHash<T>(identifier, index))
		{
			if (m_hash == nullptr)
			{
				m_hash = hash;
			}
			else
			{
				auto resources = this->m_builder.GetLocalResources();
				auto newHash = resources->template AllocateTemporary<PTX::UInt32Type>();
				this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(newHash, m_hash, hash));
				m_hash = newHash;
			}
		}
	}

private:
	template<class T>
	PTX::Register<PTX::UInt32Type> *GenerateHash(const HorseIR::Identifier *identifier, unsigned int cellIndex = 0)
	{
		auto comparisonIndex = (m_comparisonOperations.size() == 1) ? 0 : cellIndex;
		if (m_comparisonOperations.at(comparisonIndex) != ComparisonOperation::Equal)
		{
			return nullptr;
		}

		OperandGenerator<B, T> operandGenerator(this->m_builder);
		operandGenerator.SetBoundsCheck(false);
		auto value = operandGenerator.GenerateRegister(identifier, OperandGenerator<B, T>::LoadKind::Vector, cellIndex);

		// murmur3 hash: https://github.com/aappleby/smhasher/
		//
		//   k ^= k >> 16;
		//   k *= 0x85ebca6b; // 2246822507
		//   k ^= k >> 13;
		//   k *= 0xc2b2ae35; // 3266489909
		//   k ^= k >> 16;

		auto convertedValue = ConversionGenerator::ConvertSource<PTX::UInt32Type, T>(this->m_builder, value);

		auto resources = this->m_builder.GetLocalResources();
		auto hash = resources->template AllocateTemporary<PTX::UInt32Type>();
		auto bitHash = new PTX::Bit32RegisterAdapter<PTX::UIntType>(hash);
		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(hash, convertedValue));

		auto hash1 = resources->template AllocateTemporary<PTX::Bit32Type>();
		auto hash2 = resources->template AllocateTemporary<PTX::Bit32Type>();
		auto hash3 = resources->template AllocateTemporary<PTX::Bit32Type>();

		this->m_builder.AddStatement(new PTX::ShiftRightInstruction<PTX::Bit32Type>(hash1, bitHash, new PTX::UInt32Value(16)));
		this->m_builder.AddStatement(new PTX::XorInstruction<PTX::Bit32Type>(bitHash, bitHash, hash1));

		this->m_builder.AddStatement(new PTX::MultiplyInstruction<PTX::UInt32Type>(
			hash, hash, new PTX::UInt32Value(0x85ebca6b), PTX::HalfModifier<PTX::UInt32Type>::Half::Lower
		));

		this->m_builder.AddStatement(new PTX::ShiftRightInstruction<PTX::Bit32Type>(hash2, bitHash, new PTX::UInt32Value(13)));
		this->m_builder.AddStatement(new PTX::XorInstruction<PTX::Bit32Type>(bitHash, bitHash, hash2));

		this->m_builder.AddStatement(new PTX::MultiplyInstruction<PTX::UInt32Type>(
			hash, hash, new PTX::UInt32Value(0xc2b2ae35), PTX::HalfModifier<PTX::UInt32Type>::Half::Lower
		));

		this->m_builder.AddStatement(new PTX::ShiftRightInstruction<PTX::Bit32Type>(hash3, bitHash, new PTX::UInt32Value(16)));
		this->m_builder.AddStatement(new PTX::XorInstruction<PTX::Bit32Type>(bitHash, bitHash, hash3));

		return hash;
	}

	std::vector<ComparisonOperation> m_comparisonOperations;
	PTX::Register<PTX::UInt32Type> *m_hash = nullptr;
};

template<PTX::Bits B, class K>
class InternalHashEqualGenerator : public Generator, public HorseIR::ConstVisitor
{
public:
	using Generator::Generator;

	std::string Name() const override { return "InternalHashEqualGenerator"; }

	std::tuple<PTX::Register<PTX::PredicateType> *, PTX::TypedOperand<K> *, std::string> Generate(
		const HorseIR::Operand *dataOperand, const HorseIR::Operand *keyOperand, PTX::TypedOperand<PTX::UInt32Type> *slot,
		const std::vector<ComparisonOperation>& comparisonOperations
	)
	{
		m_comparisonOperations = comparisonOperations;
		m_keyOperand = keyOperand;
		m_slot = slot;

		m_predicate = nullptr;
		m_slotValue = nullptr;
		dataOperand->Accept(*this);
		return {m_predicate, m_slotValue, m_slotIdentifier};
	}

	void Visit(const HorseIR::Identifier *identifier) override
	{
		DispatchType(*this, identifier->GetType(), identifier);
	}

	void Visit(const HorseIR::Literal *literal) override
	{
		Error("literal data for internal equals");
	}

	template<class T>
	void GenerateVector(const HorseIR::Identifier *identifier)
	{
		m_predicate = GeneratePredicate<T>(identifier);
	}

	template<class T>
	void GenerateList(const HorseIR::Identifier *identifier)
	{
		const auto& inputOptions = this->m_builder.GetInputOptions();
		const auto parameter = inputOptions.Parameters.at(identifier->GetSymbol());
		const auto shape = inputOptions.ParameterShapes.at(parameter);

		if (const auto listShape = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::ListShape>(shape))
		{
			if (const auto size = HorseIR::Analysis::ShapeUtils::GetSize<HorseIR::Analysis::Shape::ConstantSize>(listShape->GetListSize()))
			{
				for (auto index = 0u; index < size->GetValue(); ++index)
				{
					GenerateTuple<T>(index, identifier);
				}
				return;
			}
		}
		Error("non-constant cell count");
	}

	template<class T>
	void GenerateTuple(unsigned int index, const HorseIR::Identifier *identifier)
	{
		auto predicate = GeneratePredicate<T>(identifier, index);
		if (m_predicate == nullptr)
		{
			m_predicate = predicate;
		}
		else
		{
			auto resources = this->m_builder.GetLocalResources();
			auto newPredicate = resources->template AllocateTemporary<PTX::PredicateType>();
			this->m_builder.AddStatement(new PTX::AndInstruction<PTX::PredicateType>(newPredicate, m_predicate, predicate));
			m_predicate = newPredicate;
		}
	}

private:
	template<class T>
	PTX::Register<PTX::PredicateType> *GeneratePredicate(const HorseIR::Identifier *dataOperand, unsigned int cellIndex = 0)
	{
		OperandGenerator<B, T> operandGenerator(this->m_builder);
		operandGenerator.SetBoundsCheck(false);

		auto slotIdentifier = this->m_builder.UniqueIdentifier("slot");
		if (cellIndex == 0)
		{
			m_slotIdentifier = slotIdentifier;
		}

		auto value = operandGenerator.GenerateOperand(dataOperand, OperandGenerator<B, T>::LoadKind::Vector, cellIndex);
		auto slotValue = operandGenerator.GenerateOperand(m_keyOperand, m_slot, slotIdentifier, cellIndex);

		if (cellIndex == 0)
		{
			if constexpr(std::is_same<T, K>::value)
			{
				m_slotValue = slotValue;
			}
			else
			{
				Error("inconsistent first cell type");
			}
		}

		auto resources = this->m_builder.GetLocalResources();
		auto predicate = resources->template AllocateTemporary<PTX::PredicateType>();

		auto comparisonIndex = (m_comparisonOperations.size() == 1) ? 0 : cellIndex;

		ComparisonGenerator<B, PTX::PredicateType> comparisonGenerator(this->m_builder, m_comparisonOperations.at(comparisonIndex));
		comparisonGenerator.Generate(predicate, value, slotValue);

		return predicate;
	}

	std::vector<ComparisonOperation> m_comparisonOperations;

	PTX::TypedOperand<PTX::UInt32Type> *m_slot = nullptr;
	std::string m_slotIdentifier;
	const HorseIR::Operand *m_keyOperand = nullptr;

	PTX::Register<PTX::PredicateType> *m_predicate = nullptr;
	PTX::TypedOperand<K> *m_slotValue = nullptr;
};

template<PTX::Bits B>
class InternalHashEmptyGenerator : public Generator, public HorseIR::ConstVisitor
{
public:
	using Generator::Generator;

	std::string Name() const override { return "InternalHashEmptyGenerator"; }

	PTX::Register<PTX::PredicateType> *Generate(const HorseIR::Operand *keyOperand, PTX::TypedOperand<PTX::UInt32Type> *slot, const std::string& slotIdentifier)
	{
		m_slot = slot;
		m_slotIdentifier = slotIdentifier;
		m_predicate = nullptr;

		keyOperand->Accept(*this);

		return m_predicate;
	}

	void Visit(const HorseIR::Identifier *identifier) override
	{
		DispatchType(*this, identifier->GetType(), identifier);
	}

	void Visit(const HorseIR::Literal *literal) override
	{
		Error("literal data for internal empty");
	}

	template<class T>
	void GenerateVector(const HorseIR::Identifier *identifier)
	{
		m_predicate = GeneratePredicate<T>(identifier);
	}

	template<class T>
	void GenerateList(const HorseIR::Identifier *identifier)
	{
		const auto& inputOptions = this->m_builder.GetInputOptions();
		const auto parameter = inputOptions.Parameters.at(identifier->GetSymbol());
		const auto shape = inputOptions.ParameterShapes.at(parameter);

		if (const auto listShape = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::ListShape>(shape))
		{
			if (const auto size = HorseIR::Analysis::ShapeUtils::GetSize<HorseIR::Analysis::Shape::ConstantSize>(listShape->GetListSize()))
			{
				for (auto index = 0u; index < size->GetValue(); ++index)
				{
					GenerateTuple<T>(index, identifier);
				}
				return;
			}
		}
		Error("non-constant cell count");
	}

	template<class T>
	void GenerateTuple(unsigned int index, const HorseIR::Identifier *identifier)
	{
		if (index == 0)
		{
			m_predicate = GeneratePredicate<T>(identifier);
		}
	}

private:
	template<class T>
	PTX::Register<PTX::PredicateType> *GeneratePredicate(const HorseIR::Identifier *keyOperand)
	{
		OperandGenerator<B, T> operandGenerator(this->m_builder);
		operandGenerator.SetBoundsCheck(false);

		auto slotValue = operandGenerator.GenerateOperand(keyOperand, m_slot, m_slotIdentifier);
		auto emptyValue = new PTX::Value<T>(std::numeric_limits<typename T::SystemType>::max());

		auto resources = this->m_builder.GetLocalResources();
		auto emptyPredicate = resources->template AllocateTemporary<PTX::PredicateType>();

		ComparisonGenerator<B, PTX::PredicateType> comparisonGenerator(this->m_builder, ComparisonOperation::NotEqual);
		comparisonGenerator.Generate(emptyPredicate, emptyValue, slotValue);

		return emptyPredicate;
	}

	std::string m_slotIdentifier;
	PTX::TypedOperand<PTX::UInt32Type> *m_slot = nullptr;
	PTX::Register<PTX::PredicateType> *m_predicate = nullptr;
};

}
}
