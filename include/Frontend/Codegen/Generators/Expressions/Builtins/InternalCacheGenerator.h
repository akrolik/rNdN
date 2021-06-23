#pragma once

#include "HorseIR/Traversal/ConstVisitor.h"
#include "Frontend/Codegen/Generators/Generator.h"

#include "Frontend/Codegen/Generators/Expressions/OperandGenerator.h"
#include "Frontend/Codegen/Generators/Indexing/ThreadIndexGenerator.h"
#include "Frontend/Codegen/Generators/Indexing/AddressGenerator.h"
#include "Frontend/Codegen/Generators/Synchronization/BarrierGenerator.h"

#include "HorseIR/Analysis/Shape/Shape.h"
#include "HorseIR/Analysis/Shape/ShapeUtils.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/Tree/Tree.h"

namespace Frontend {
namespace Codegen {

template<PTX::Bits B, unsigned int CACHE_SIZE = 1024u, unsigned int N = 1>
class InternalCacheGenerator_Load : public Generator, public HorseIR::ConstVisitor
{
public:
	using Generator::Generator; 

	std::string Name() const override { return "InternalCacheGenerator_Load"; }

	bool GetBoundsCheck() const { return m_boundsCheck; }
	void SetBoundsCheck(bool boundsCheck) { m_boundsCheck = boundsCheck; }
	
	bool GetSynchronize() const { return m_synchronize; }
	void SetSynchronize(bool synchronize) { m_synchronize = synchronize; }

	void Generate(const HorseIR::Operand *data, PTX::TypedOperand<PTX::UInt32Type> *index)
	{
		this->m_builder.SetBlockSize(CACHE_SIZE);

		m_index = index;
		data->Accept(*this);
	}

	void Visit(const HorseIR::Identifier *data) override
	{
		Generate(data, m_index, data->GetType());
	}

	void Visit(const HorseIR::Literal *data) override
	{
		Error("literal data");
	}

	void Generate(const HorseIR::Identifier *data, PTX::TypedOperand<PTX::UInt32Type> *index, const HorseIR::Type *type)
	{
		// Load data into shared memory

		DispatchType(*this, type, data, index);

		// Synchronize shared memory

		if (m_synchronize)
		{
			BarrierGenerator<B> barrierGenerator(this->m_builder);
			barrierGenerator.Generate();
		}
	}

	template<class T>
	void GenerateVector(const HorseIR::Identifier *data, PTX::TypedOperand<PTX::UInt32Type> *index)
	{
		GenerateCache<T>(data, index);
	}

	template<class T>
	void GenerateList(const HorseIR::Identifier *data, PTX::TypedOperand<PTX::UInt32Type> *index)
	{
		const auto& inputOptions = this->m_builder.GetInputOptions();
		const auto parameter = inputOptions.Parameters.at(data->GetSymbol());
		const auto shape = inputOptions.ParameterShapes.at(parameter);

		if (const auto listShape = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::ListShape>(shape))
		{
			if (const auto size = HorseIR::Analysis::ShapeUtils::GetSize<HorseIR::Analysis::Shape::ConstantSize>(listShape->GetListSize()))
			{
				for (auto cellIndex = 0u; cellIndex < size->GetValue(); ++cellIndex)
				{
					GenerateCache<T>(data, index, true, cellIndex);
				}
				return;
			}
		}
		Error("non-constant cell count");
	}

	template<class T>
	void GenerateTuple(unsigned int cellIndex, const HorseIR::Identifier *data, PTX::TypedOperand<PTX::UInt32Type> *index)
	{
		if (!this->m_builder.GetInputOptions().IsVectorGeometry())
		{
			Error("tuple-in-list");
		}
		GenerateCache<T>(data, index, true, cellIndex);
	}

private:
	template<class T>
	void GenerateCache(const HorseIR::Identifier *data, PTX::TypedOperand<PTX::UInt32Type> *index, bool isCell = false, unsigned int cellIndex = 0)
	{
		if constexpr(std::is_same<T, PTX::PredicateType>::value)
		{
			GenerateCache<PTX::Int8Type>(data, index, isCell, cellIndex);
		}
		else
		{
			// Allocate shared memory space used for data cache

			auto resources = this->m_builder.GetLocalResources();
			auto kernelResources = this->m_builder.GetKernelResources();

			auto cacheName = data->GetName() + "_cache" + ((isCell) ? std::to_string(cellIndex) : "");
			auto s_cache = new PTX::ArrayVariableAdapter<T, CACHE_SIZE * N, PTX::SharedSpace>(
				kernelResources->template AllocateSharedVariable<PTX::ArrayType<T, CACHE_SIZE * N>>(cacheName)
			);

			// Load the cache from global memory and store in shared (cell index default to zero for vector)
			//
			// When N > 1, load the data in CACHE_SIZE chunks (assumes the index is correctly offset)
			//
			//    s_data[threadIdx.x                 ] = data[index                 ];
			//    s_data[threadIdx.x + CACHE_SIZE    ] = data[index + CACHE_SIZE    ];
			//    ...
			//    s_data[threadIdx.x + CACHE_SIZE * N] = data[index + CACHE_SIZE * N];

			ThreadIndexGenerator<B> threadGenerator(this->m_builder);
			auto localIndex = threadGenerator.GenerateLocalIndex();

			AddressGenerator<B, T> addressGenerator(this->m_builder);
			auto s_cacheAddress = addressGenerator.GenerateAddress(s_cache, localIndex);

			for (auto i = 0; i < N; ++i)
			{
				OperandGenerator<B, T> operandGenerator(this->m_builder);
				operandGenerator.SetBoundsCheck(m_boundsCheck);
				auto value = operandGenerator.GenerateRegister(data, index, this->m_builder.UniqueIdentifier("cache"), cellIndex);

				this->m_builder.AddStatement(new PTX::StoreInstruction<B, T, PTX::SharedSpace>(s_cacheAddress, value));

				if (i + 1 < N)
				{
					s_cacheAddress = s_cacheAddress->CreateOffsetAddress(CACHE_SIZE);

					auto newIndex = resources->template AllocateTemporary<PTX::UInt32Type>();
					this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(newIndex, index, new PTX::UInt32Value(CACHE_SIZE)));
					index = newIndex;
				}
			}
		}
	}

	bool m_boundsCheck = true;
	bool m_synchronize = true;
	PTX::TypedOperand<PTX::UInt32Type> *m_index = nullptr;
};

template<PTX::Bits B, unsigned int CACHE_SIZE = 1024u, unsigned int N = 1>
class InternalCacheGenerator_Store : public Generator, public HorseIR::ConstVisitor
{
public:
	using Generator::Generator; 

	std::string Name() const override { return "InternalCacheGenerator_Load"; }

	void Generate(const HorseIR::Operand *data, PTX::TypedOperand<PTX::UInt32Type> *index)
	{
		m_index = index;
		data->Accept(*this);
	}

	void Visit(const HorseIR::Identifier *data) override
	{
		Generate(data, m_index, data->GetType());
	}

	void Visit(const HorseIR::Literal *data) override
	{
		Error("literal data");
	}

	void Generate(const HorseIR::Identifier *data, PTX::TypedOperand<PTX::UInt32Type> *index, const HorseIR::Type *type)
	{
		// Store data from shared memory

		DispatchType(*this, type, data, index);
	}

	template<class T>
	void GenerateVector(const HorseIR::Identifier *data, PTX::TypedOperand<PTX::UInt32Type> *index)
	{
		GenerateCache<T>(data, index);
	}

	template<class T>
	void GenerateList(const HorseIR::Identifier *data, PTX::TypedOperand<PTX::UInt32Type> *index)
	{
		const auto& inputOptions = this->m_builder.GetInputOptions();
		const auto parameter = inputOptions.Parameters.at(data->GetSymbol());
		const auto shape = inputOptions.ParameterShapes.at(parameter);

		if (const auto listShape = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::ListShape>(shape))
		{
			if (const auto size = HorseIR::Analysis::ShapeUtils::GetSize<HorseIR::Analysis::Shape::ConstantSize>(listShape->GetListSize()))
			{
				for (auto cellIndex = 0u; cellIndex < size->GetValue(); ++cellIndex)
				{
					GenerateTuple<T>(cellIndex, data, index);
				}
				return;
			}
		}
		Error("non-constant cell count");
	}

	template<class T>
	void GenerateTuple(unsigned int cellIndex, const HorseIR::Identifier *data, PTX::TypedOperand<PTX::UInt32Type> *index)
	{
		if (!this->m_builder.GetInputOptions().IsVectorGeometry())
		{
			Error("tuple-in-list");
		}
		GenerateCache<T>(data, index, true, cellIndex);
	}

private:
	template<class T>
	void GenerateCache(const HorseIR::Identifier *data, PTX::TypedOperand<PTX::UInt32Type> *index, bool isCell = false, unsigned int cellIndex = 0)
	{
		if constexpr(std::is_same<T, PTX::PredicateType>::value)
		{
			GenerateCache<PTX::Int8Type>(data, index, isCell, cellIndex);
		}
		else
		{
			// Allocate shared memory space used for data cache

			auto resources = this->m_builder.GetLocalResources();
			auto kernelResources = this->m_builder.GetKernelResources();

			auto cacheName = data->GetName() + "_cache" + ((isCell) ? std::to_string(cellIndex) : "");
			auto s_cache = new PTX::ArrayVariableAdapter<T, CACHE_SIZE * N, PTX::SharedSpace>(
				kernelResources->template GetSharedVariable<PTX::ArrayType<T, CACHE_SIZE * N>>(cacheName)
			);

			// Load the cache from shared memory and store in global (cell index default to zero for vector)
			//
			// When N > 1, store the data in CACHE_SIZE chunks (assumes the index is correctly offset)
			//
			//    data[index                 ] = s_data[threadIdx.x                 ];
			//    data[index + CACHE_SIZE    ] = s_data[threadIdx.x + CACHE_SIZE    ];
			//    ...
			//    data[index + CACHE_SIZE * N] = s_data[threadIdx.x + CACHE_SIZE * N];

			ThreadIndexGenerator<B> threadGenerator(this->m_builder);
			auto localIndex = threadGenerator.GenerateLocalIndex();

			AddressGenerator<B, T> addressGenerator(this->m_builder);
			auto s_cacheAddress = addressGenerator.GenerateAddress(s_cache, localIndex);

			for (auto i = 0; i < N; ++i)
			{
				if (isCell)
				{
					auto parameter = kernelResources->template GetParameter<PTX::PointerType<B, PTX::PointerType<B, T, PTX::GlobalSpace>>>(NameUtils::VariableName(data));

					auto value = resources->template AllocateTemporary<T>();
					auto globalAddress = addressGenerator.GenerateAddress(parameter, cellIndex, index);

					this->m_builder.AddStatement(new PTX::LoadInstruction<B, T, PTX::SharedSpace>(value, s_cacheAddress));
					this->m_builder.AddStatement(new PTX::StoreInstruction<B, T, PTX::GlobalSpace>(globalAddress, value));
				}
				else
				{
					auto parameter = kernelResources->template GetParameter<PTX::PointerType<B, T>>(NameUtils::VariableName(data));

					auto value = resources->template AllocateTemporary<T>();
					auto globalAddress = addressGenerator.GenerateAddress(parameter, index);

					this->m_builder.AddStatement(new PTX::LoadInstruction<B, T, PTX::SharedSpace>(value, s_cacheAddress));
					this->m_builder.AddStatement(new PTX::StoreInstruction<B, T, PTX::GlobalSpace>(globalAddress, value));
				}

				if (i + 1 < N)
				{
					s_cacheAddress = s_cacheAddress->CreateOffsetAddress(CACHE_SIZE);

					auto newIndex = resources->template AllocateTemporary<PTX::UInt32Type>();
					this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(newIndex, index, new PTX::UInt32Value(CACHE_SIZE)));
					index = newIndex;
				}
			}
		}
	}

	PTX::TypedOperand<PTX::UInt32Type> *m_index = nullptr;
};

}
}
