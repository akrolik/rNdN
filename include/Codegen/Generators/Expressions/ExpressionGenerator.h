#pragma once

#include "HorseIR/Traversal/ConstVisitor.h"
#include "Codegen/Generators/Generator.h"

#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/TypeUtils.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/TypeDispatch.h"
#include "Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "Codegen/Generators/Expressions/Builtins/BinaryGenerator.h"
#include "Codegen/Generators/Expressions/Builtins/ComparisonGenerator.h"
#include "Codegen/Generators/Expressions/Builtins/ExternalBinaryGenerator.h"
#include "Codegen/Generators/Expressions/Builtins/ExternalUnaryGenerator.h"
#include "Codegen/Generators/Expressions/Builtins/RoundingGenerator.h"
#include "Codegen/Generators/Expressions/Builtins/UnaryGenerator.h"

#include "Codegen/Generators/Expressions/Builtins/UniqueGenerator.h"
#include "Codegen/Generators/Expressions/Builtins/RangeGenerator.h"
#include "Codegen/Generators/Expressions/Builtins/GroupGenerator.h"
#include "Codegen/Generators/Expressions/Builtins/WhereGenerator.h"

#include "Codegen/Generators/Expressions/Builtins/AppendGenerator.h"
#include "Codegen/Generators/Expressions/Builtins/CompressionGenerator.h"
#include "Codegen/Generators/Expressions/Builtins/IndexOfGenerator.h"
#include "Codegen/Generators/Expressions/Builtins/LimitGenerator.h"
#include "Codegen/Generators/Expressions/Builtins/MemberGenerator.h"
#include "Codegen/Generators/Expressions/Builtins/OrderInitGenerator.h"
#include "Codegen/Generators/Expressions/Builtins/OrderGenerator.h"
#include "Codegen/Generators/Expressions/Builtins/VectorGenerator.h"

#include "Codegen/Generators/Expressions/Builtins/ReductionGenerator.h"

#include "Codegen/Generators/Expressions/Builtins/RazeGenerator.h"

#include "Codegen/Generators/Expressions/Builtins/DateGenerator.h"

#include "Codegen/Generators/Expressions/Builtins/JoinCountGenerator.h"
#include "Codegen/Generators/Expressions/Builtins/JoinGenerator.h"

#include "Codegen/Generators/Expressions/Builtins/IndexedReadGenerator.h"
#include "Codegen/Generators/Expressions/Builtins/IndexedWriteGenerator.h"

#include "PTX/PTX.h"

namespace Codegen {

template<PTX::Bits B>
class ExpressionGenerator : public HorseIR::ConstVisitor, public Generator
{
public:
	using Generator::Generator;

	std::string Name() const override { return "ExpressionGenerator"; }

	void Generate(const std::vector<HorseIR::LValue *>& targets, const HorseIR::Expression *expression)
	{
		m_targets = targets;
		expression->Accept(*this);
	}

	void Generate(const HorseIR::Expression *expression)
	{
		m_targets.clear();
		expression->Accept(*this);
	}

	void Visit(const HorseIR::CallExpression *call) override
	{
		auto function = call->GetFunctionLiteral()->GetFunction();
		Generate(function, call->GetTypes(), call->GetArguments());
	}

	void Generate(const HorseIR::FunctionDeclaration *function, const std::vector<HorseIR::Type *>& returnTypes, const std::vector<HorseIR::Operand *>& arguments)
	{
		switch (function->GetKind())
		{
			case HorseIR::FunctionDeclaration::Kind::Builtin:
			{
				Generate(static_cast<const HorseIR::BuiltinFunction *>(function), returnTypes, arguments);
				break;
			}
			case HorseIR::FunctionDeclaration::Kind::Definition:
			{
				Error("user defined function @" + function->GetName());
			}
		}
	}

	void Generate(const HorseIR::BuiltinFunction *function, const std::vector<HorseIR::Type *>& returnTypes, const std::vector<HorseIR::Operand *>& arguments)
	{
		switch (function->GetPrimitive())
		{
			case HorseIR::BuiltinFunction::Primitive::GPUOrderInit:
			{
				OrderInitGenerator<B> generator(this->m_builder);
				generator.Generate(m_targets, arguments);
				break;
			}
			case HorseIR::BuiltinFunction::Primitive::GPUOrder:
			{
				OrderGenerator<B> generator(this->m_builder);
				generator.Generate(arguments);
				break;
			}
			case HorseIR::BuiltinFunction::Primitive::GPUGroup:
			{
				GroupGenerator<B> generator(this->m_builder);
				generator.Generate(m_targets, arguments);
				break;
			}
			case HorseIR::BuiltinFunction::Primitive::GPUUnique:
			{
				UniqueGenerator<B> generator(this->m_builder);
				generator.Generate(m_targets, arguments);
				break;
			}
			case HorseIR::BuiltinFunction::Primitive::GPUJoinCount:
			{
				JoinCountGenerator<B> generator(this->m_builder);
				generator.Generate(m_targets, arguments);
				break;
			}
			case HorseIR::BuiltinFunction::Primitive::GPUJoin:
			{
				JoinGenerator<B> generator(this->m_builder);
				generator.Generate(m_targets.at(0), arguments);
				break;
			}
			default:
			{
				if (m_targets.size() != 1)
				{
					Error("builtin function @" + function->GetName() + " with multiple targets");
				}
				DispatchType(*this, HorseIR::TypeUtils::GetSingleType(returnTypes), function, returnTypes, arguments);
			}
		}
	}

	template<class T>
	void GenerateVector(const HorseIR::BuiltinFunction *function, const std::vector<HorseIR::Type *>& returnTypes, const std::vector<HorseIR::Operand *>& arguments)
	{
		switch (function->GetPrimitive())
		{
			case HorseIR::BuiltinFunction::Primitive::Each: 
			case HorseIR::BuiltinFunction::Primitive::EachItem: 
			case HorseIR::BuiltinFunction::Primitive::EachLeft: 
			case HorseIR::BuiltinFunction::Primitive::EachRight: 
			{
				// List functions apply the underlying function. We ensure correct geometry through
				// the kernel setup (threads and data loading)

				const auto l_function = HorseIR::TypeUtils::GetType<HorseIR::FunctionType>(arguments.at(0)->GetType())->GetFunctionDeclaration();

				std::vector<HorseIR::Operand *> l_arguments;
				l_arguments.insert(std::begin(l_arguments), std::begin(arguments) + 1, std::end(arguments));

				Generate(l_function, returnTypes, l_arguments);
				break;
			}
			default:
			{
				BuiltinGenerator<B, T> *generator = GetBuiltinGenerator<T>(function);
				if (generator == nullptr)
				{
					Error("buitin function @" + function->GetName());
				}
				generator->Generate(m_targets.at(0), arguments);
				delete generator;
				break;
			}
		}
	}

	template<typename T>
	BuiltinGenerator<B, T> *GetBuiltinGenerator(const HorseIR::BuiltinFunction *function)
	{
		switch (function->GetPrimitive())
		{
			// Unary
			case HorseIR::BuiltinFunction::Primitive::Absolute:
				return new UnaryGenerator<B, T>(this->m_builder, UnaryOperation::Absolute);
			case HorseIR::BuiltinFunction::Primitive::Negate:
				return new UnaryGenerator<B, T>(this->m_builder, UnaryOperation::Negate);
			case HorseIR::BuiltinFunction::Primitive::Ceiling:
				return new RoundingGenerator<B, T>(this->m_builder, RoundingOperation::Ceiling);
			case HorseIR::BuiltinFunction::Primitive::Floor:
				return new RoundingGenerator<B, T>(this->m_builder, RoundingOperation::Floor);
			case HorseIR::BuiltinFunction::Primitive::Round:
				return new RoundingGenerator<B, T>(this->m_builder, RoundingOperation::Nearest);
			case HorseIR::BuiltinFunction::Primitive::Reciprocal:
				return new UnaryGenerator<B, T>(this->m_builder, UnaryOperation::Reciprocal);
			case HorseIR::BuiltinFunction::Primitive::Sign:
				return new ComparisonGenerator<B, T>(this->m_builder, ComparisonOperation::Sign);
			case HorseIR::BuiltinFunction::Primitive::Pi:
				return new UnaryGenerator<B, T>(this->m_builder, UnaryOperation::Pi);
			case HorseIR::BuiltinFunction::Primitive::Not:
				return new UnaryGenerator<B, T>(this->m_builder, UnaryOperation::Not);
			case HorseIR::BuiltinFunction::Primitive::Logarithm:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::Logarithm);
			case HorseIR::BuiltinFunction::Primitive::Logarithm2:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::Logarithm2);
			case HorseIR::BuiltinFunction::Primitive::Logarithm10:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::Logarithm10);
			case HorseIR::BuiltinFunction::Primitive::SquareRoot:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::SquareRoot);
			case HorseIR::BuiltinFunction::Primitive::Exponential:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::Exponential);
			case HorseIR::BuiltinFunction::Primitive::Cosine:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::Cosine);
			case HorseIR::BuiltinFunction::Primitive::Sine:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::Sine);
			case HorseIR::BuiltinFunction::Primitive::Tangent:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::Tangent);
			case HorseIR::BuiltinFunction::Primitive::InverseCosine:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::InverseCosine);
			case HorseIR::BuiltinFunction::Primitive::InverseSine:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::InverseSine);
			case HorseIR::BuiltinFunction::Primitive::InverseTangent:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::InverseTangent);
			case HorseIR::BuiltinFunction::Primitive::HyperbolicCosine:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::HyperbolicCosine);
			case HorseIR::BuiltinFunction::Primitive::HyperbolicSine:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::HyperbolicSine);
			case HorseIR::BuiltinFunction::Primitive::HyperbolicTangent:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::HyperbolicTangent);
			case HorseIR::BuiltinFunction::Primitive::HyperbolicInverseCosine:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::HyperbolicInverseCosine);
			case HorseIR::BuiltinFunction::Primitive::HyperbolicInverseSine:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::HyperbolicInverseSine);
			case HorseIR::BuiltinFunction::Primitive::HyperbolicInverseTangent:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::HyperbolicInverseTangent);

			// Binary
			case HorseIR::BuiltinFunction::Primitive::Less:
				return new ComparisonGenerator<B, T>(this->m_builder, ComparisonOperation::Less);
			case HorseIR::BuiltinFunction::Primitive::Greater:
				return new ComparisonGenerator<B, T>(this->m_builder, ComparisonOperation::Greater);
			case HorseIR::BuiltinFunction::Primitive::LessEqual:
				return new ComparisonGenerator<B, T>(this->m_builder, ComparisonOperation::LessEqual);
			case HorseIR::BuiltinFunction::Primitive::GreaterEqual:
				return new ComparisonGenerator<B, T>(this->m_builder, ComparisonOperation::GreaterEqual);
			case HorseIR::BuiltinFunction::Primitive::Equal:
				return new ComparisonGenerator<B, T>(this->m_builder, ComparisonOperation::Equal);
			case HorseIR::BuiltinFunction::Primitive::NotEqual:
				return new ComparisonGenerator<B, T>(this->m_builder, ComparisonOperation::NotEqual);
			case HorseIR::BuiltinFunction::Primitive::Plus:
				return new BinaryGenerator<B, T>(this->m_builder, BinaryOperation::Plus);
			case HorseIR::BuiltinFunction::Primitive::Minus:
				return new BinaryGenerator<B, T>(this->m_builder, BinaryOperation::Minus);
			case HorseIR::BuiltinFunction::Primitive::Multiply:
				return new BinaryGenerator<B, T>(this->m_builder, BinaryOperation::Multiply);
			case HorseIR::BuiltinFunction::Primitive::Divide:
				return new BinaryGenerator<B, T>(this->m_builder, BinaryOperation::Divide);
			case HorseIR::BuiltinFunction::Primitive::Power:
				return new ExternalBinaryGenerator<B, T>(this->m_builder, ExternalBinaryOperation::Power);
			case HorseIR::BuiltinFunction::Primitive::LogarithmBase:
				return new ExternalBinaryGenerator<B, T>(this->m_builder, ExternalBinaryOperation::Logarithm);
			case HorseIR::BuiltinFunction::Primitive::Modulo:
				return new ExternalBinaryGenerator<B, T>(this->m_builder, ExternalBinaryOperation::Modulo);
			case HorseIR::BuiltinFunction::Primitive::And:
				return new BinaryGenerator<B, T>(this->m_builder, BinaryOperation::And);
			case HorseIR::BuiltinFunction::Primitive::Or:
				return new BinaryGenerator<B, T>(this->m_builder, BinaryOperation::Or);
			case HorseIR::BuiltinFunction::Primitive::Nand:
				return new BinaryGenerator<B, T>(this->m_builder, BinaryOperation::Nand);
			case HorseIR::BuiltinFunction::Primitive::Nor:
				return new BinaryGenerator<B, T>(this->m_builder, BinaryOperation::Nor);
			case HorseIR::BuiltinFunction::Primitive::Xor:
				return new BinaryGenerator<B, T>(this->m_builder, BinaryOperation::Xor);

			// Algebraic Unary
			case HorseIR::BuiltinFunction::Primitive::Range:
				return new RangeGenerator<B, T>(this->m_builder);
			case HorseIR::BuiltinFunction::Primitive::Where:
				return new WhereGenerator<B, T>(this->m_builder);

			// Algebraic Binary
			case HorseIR::BuiltinFunction::Primitive::Append:
				return new AppendGenerator<B, T>(this->m_builder);
			case HorseIR::BuiltinFunction::Primitive::Compress:
				return new CompressionGenerator<B, T>(this->m_builder);
			case HorseIR::BuiltinFunction::Primitive::IndexOf:
				return new IndexOfGenerator<B, T>(this->m_builder);
			case HorseIR::BuiltinFunction::Primitive::Take:
				return new LimitGenerator<B, T>(this->m_builder, LimitOperation::Take);
			case HorseIR::BuiltinFunction::Primitive::Drop:
				return new LimitGenerator<B, T>(this->m_builder, LimitOperation::Drop);
			case HorseIR::BuiltinFunction::Primitive::Member:
				return new MemberGenerator<B, T>(this->m_builder);
			case HorseIR::BuiltinFunction::Primitive::Vector:
				return new VectorGenerator<B, T>(this->m_builder);

			// Reduction
			case HorseIR::BuiltinFunction::Primitive::Length:
				return new ReductionGenerator<B, T>(this->m_builder, ReductionOperation::Length);
			case HorseIR::BuiltinFunction::Primitive::Sum:
				return new ReductionGenerator<B, T>(this->m_builder, ReductionOperation::Sum);
			case HorseIR::BuiltinFunction::Primitive::Average:
				return new ReductionGenerator<B, T>(this->m_builder, ReductionOperation::Average);
			case HorseIR::BuiltinFunction::Primitive::Minimum:
				return new ReductionGenerator<B, T>(this->m_builder, ReductionOperation::Minimum);
			case HorseIR::BuiltinFunction::Primitive::Maximum:
				return new ReductionGenerator<B, T>(this->m_builder, ReductionOperation::Maximum);

			// List
			case HorseIR::BuiltinFunction::Primitive::Raze:
				return new RazeGenerator<B, T>(this->m_builder);

			// Date
			case HorseIR::BuiltinFunction::Primitive::Date:
				return new DateGenerator<B, T>(this->m_builder, DateOperation::Date);
			case HorseIR::BuiltinFunction::Primitive::DateYear:
				return new DateGenerator<B, T>(this->m_builder, DateOperation::DateYear);
			case HorseIR::BuiltinFunction::Primitive::DateMonth:
				return new DateGenerator<B, T>(this->m_builder, DateOperation::DateMonth);
			case HorseIR::BuiltinFunction::Primitive::DateDay:
				return new DateGenerator<B, T>(this->m_builder, DateOperation::DateDay);
			case HorseIR::BuiltinFunction::Primitive::Time:
				return new DateGenerator<B, T>(this->m_builder, DateOperation::Time);
			case HorseIR::BuiltinFunction::Primitive::TimeHour:
				return new DateGenerator<B, T>(this->m_builder, DateOperation::TimeHour);
			case HorseIR::BuiltinFunction::Primitive::TimeMinute:
				return new DateGenerator<B, T>(this->m_builder, DateOperation::TimeMinute);
			case HorseIR::BuiltinFunction::Primitive::TimeSecond:
				return new DateGenerator<B, T>(this->m_builder, DateOperation::TimeSecond);
			case HorseIR::BuiltinFunction::Primitive::TimeMillisecond:
				return new DateGenerator<B, T>(this->m_builder, DateOperation::TimeMillisecond);

			// Indexing
			case HorseIR::BuiltinFunction::Primitive::Index:
				return new IndexedReadGenerator<B, T>(this->m_builder);
			case HorseIR::BuiltinFunction::Primitive::IndexAssignment:
				return new IndexedWriteGenerator<B, T>(this->m_builder);
		}
		return nullptr;
	}

	template<class T>
	void GenerateList(const HorseIR::BuiltinFunction *function, const std::vector<HorseIR::Type *>& returnTypes, const std::vector<HorseIR::Operand *>& arguments)
	{
		if (this->m_builder.GetInputOptions().IsVectorGeometry())
		{
			Error("list-in-vector function @" + function->GetName());
		}

		// Lists are handled by the vector code through a projection

		GenerateVector<T>(function, returnTypes, arguments);
	}

	template<class T>
	void GenerateTuple(unsigned int index, const HorseIR::BuiltinFunction *function, const std::vector<HorseIR::Type *>& returnTypes, const std::vector<HorseIR::Operand *>& arguments)
	{
		Error("list-in-vector function @" + function->GetName());
	}

private:
	std::vector<HorseIR::LValue *> m_targets;
};

}
