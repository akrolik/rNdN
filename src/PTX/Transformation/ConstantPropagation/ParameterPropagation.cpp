#include "PTX/Transformation/ConstantPropagation/ParameterPropagation.h"

#include "Utils/Chrono.h"

namespace PTX {
namespace Transformation {

void ParameterPropagation::Transform(FunctionDefinition<VoidType> *function)
{
	auto timePropagation_start = Utils::Chrono::Start("Parameter propagation '" + function->GetName() + "'");

	if (auto cfg = function->GetControlFlowGraph())
	{
		cfg->LinearOrdering([&](Analysis::ControlFlowNode& block)
		{
			block->Accept(*this);
		});
	}
	else
	{
		for (auto& statement : function->GetStatements())
		{
			statement->Accept(*this);
		}
	}

	Utils::Chrono::End(timePropagation_start);
}

void ParameterPropagation::Visit(BasicBlock *block)
{
	for (auto& statement : block->GetStatements())
	{
		statement->Accept(*this);
	}
}

void ParameterPropagation::Visit(InstructionStatement *instruction)
{
	instruction->Accept(static_cast<InstructionVisitor&>(*this));
}

void ParameterPropagation::Visit(const _LoadInstruction *instruction)
{
	instruction->Dispatch(*this);
}

void ParameterPropagation::Visit(const _ConvertToAddressInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<Bits B, class T, class S, LoadSynchronization A>
void ParameterPropagation::Visit(const LoadInstruction<B, T, S, A> *instruction)
{
	if constexpr(std::is_same<S, ParameterSpace>::value)
	{
		if (auto address = dynamic_cast<const MemoryAddress<B, T, S> *>(instruction->GetAddress()))
		{
			if (address->GetOffset() == 0)
			{
				auto variable = address->GetVariable();
				m_constantOperand = new ParameterConstant<UIntType<B>>(variable->GetName());
			}
		}
	}
}

template<Bits B, class T, class S>
void ParameterPropagation::Visit(const ConvertToAddressInstruction<B, T, S> *instruction)
{
	if constexpr(std::is_same<S, GlobalSpace>::value)
	{
		auto source = instruction->GetAddress()->GetRegister()->GetName();

		const auto& definitions = m_definitions.GetAnalysisSet();
		if (auto it = definitions.find(&source); it != definitions.end())
		{
			auto& instructions = it->second;
			if (instructions->size() == 1)
			{
				auto instruction = *(instructions->begin());
				instruction->Accept(static_cast<ConstInstructionVisitor&>(*this));
			}
		}
	}
}

void ParameterPropagation::Visit(_MADWideInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<class T>
void ParameterPropagation::Visit(MADWideInstruction<T> *instruction)
{
	auto sourceC = instruction->GetSourceC();

	m_constantOperand = nullptr;
	sourceC->Accept(static_cast<ConstOperandVisitor&>(*this));

	if (m_constantOperand != nullptr)
	{
		instruction->SetSourceC(dynamic_cast<ParameterConstant<typename T::WideType> *>(m_constantOperand));
	}
}

bool ParameterPropagation::Visit(const _Register *reg)
{
	reg->Dispatch(*this);
	return false;
}

template<class T>
void ParameterPropagation::Visit(const Register<T> *reg)
{
	auto destination = reg->GetName();

	const auto& definitions = m_definitions.GetAnalysisSet();
	if (auto it = definitions.find(&destination); it != definitions.end())
	{
		auto& instructions = it->second;
		if (instructions->size() == 1)
		{
			auto instruction = *(instructions->begin());
			instruction->Accept(static_cast<ConstInstructionVisitor&>(*this));
		}
	}
}

}
}
