#include "Analysis/Helpers/GPUAnalysisHelper.h"

#include "HorseIR/Utils/TypeUtils.h"

namespace Analysis {

bool GPUAnalysisHelper::IsGPU(const HorseIR::Statement *statement)
{
	// Reset the analysis and traverse

	m_gpu = false;
	m_synchronization = Synchronization::None;
	statement->Accept(*this);
	return m_gpu;
}

bool GPUAnalysisHelper::IsSynchronized(const HorseIR::Statement *source, const HorseIR::Statement *destination, unsigned int index)
{
	// Check for synchronization between the two statements. Non-GPU links are never marked as synchronized

	// Check for synchronization on the source statement

	m_gpu = false;
	m_synchronization = Synchronization::None;

	source->Accept(*this);
	if (!m_gpu)
	{
		return false;
	}

	auto sourceSynchronization = m_synchronization;

	// Check for synchronization on the destination statement at the index

	m_index = index;
	m_gpu = false;
	m_synchronization = Synchronization::None;

	destination->Accept(*this);
	if (!m_gpu)
	{
		return false;
	}

	auto destinationSynchronization = m_synchronization;

	// Special case: reduction followed by raze. The synchronization barrier gets pushed by 1 statement

	if (sourceSynchronization & Synchronization::Reduction && destinationSynchronization & Synchronization::Raze)
	{
		return false;
	}
	return (sourceSynchronization & Synchronization::Out || destinationSynchronization & Synchronization::In);
}

void GPUAnalysisHelper::Visit(const HorseIR::DeclarationStatement *declarationS)
{
	m_gpu = true;
	m_synchronization = Synchronization::None;
}

void GPUAnalysisHelper::Visit(const HorseIR::AssignStatement *assignS)
{
	assignS->GetExpression()->Accept(*this);
}

void GPUAnalysisHelper::Visit(const HorseIR::ExpressionStatement *expressionS)
{
	expressionS->GetExpression()->Accept(*this);
}

void GPUAnalysisHelper::Visit(const HorseIR::ReturnStatement *returnS)
{
	// Explicitly disallow return from kernels, we will insert as needed

	m_gpu = false;
	m_synchronization = Synchronization::None;
}

void GPUAnalysisHelper::Visit(const HorseIR::CallExpression *call)
{
	auto [gpu, synchronization] = AnalyzeCall(call->GetFunctionLiteral()->GetFunction(), call->GetArguments(), m_index);

	m_gpu = gpu;
	m_synchronization = synchronization;
}

std::pair<bool, GPUAnalysisHelper::Synchronization> GPUAnalysisHelper::AnalyzeCall(const HorseIR::FunctionDeclaration *function, const std::vector<HorseIR::Operand *>& arguments, unsigned int index)
{
	switch (function->GetKind())
	{
		case HorseIR::FunctionDeclaration::Kind::Builtin:
			return AnalyzeCall(static_cast<const HorseIR::BuiltinFunction *>(function), arguments, index);
		case HorseIR::FunctionDeclaration::Kind::Definition:
			return AnalyzeCall(static_cast<const HorseIR::Function *>(function), arguments, index);
		default:
			Utils::Logger::LogError("Unsupported function kind");
	}
}

std::pair<bool, GPUAnalysisHelper::Synchronization> GPUAnalysisHelper::AnalyzeCall(const HorseIR::Function *function, const std::vector<HorseIR::Operand *>& arguments, unsigned int index)
{
	return {false, Synchronization::None};
}

std::pair<bool, GPUAnalysisHelper::Synchronization> GPUAnalysisHelper::AnalyzeCall(const HorseIR::BuiltinFunction *function, const std::vector<HorseIR::Operand *>& arguments, unsigned int index)
{
	switch (function->GetPrimitive())
	{
		// ---------------
		// Vector Geometry
		// ---------------

		// Unary
		case HorseIR::BuiltinFunction::Primitive::Absolute:
		case HorseIR::BuiltinFunction::Primitive::Negate:
		case HorseIR::BuiltinFunction::Primitive::Ceiling:
		case HorseIR::BuiltinFunction::Primitive::Floor:
		case HorseIR::BuiltinFunction::Primitive::Round:
		case HorseIR::BuiltinFunction::Primitive::Conjugate:
		case HorseIR::BuiltinFunction::Primitive::Reciprocal:
		case HorseIR::BuiltinFunction::Primitive::Sign:
		case HorseIR::BuiltinFunction::Primitive::Pi:
		case HorseIR::BuiltinFunction::Primitive::Not:
		case HorseIR::BuiltinFunction::Primitive::Logarithm:
		case HorseIR::BuiltinFunction::Primitive::Logarithm2:
		case HorseIR::BuiltinFunction::Primitive::Logarithm10:
		case HorseIR::BuiltinFunction::Primitive::SquareRoot:
		case HorseIR::BuiltinFunction::Primitive::Exponential:
		case HorseIR::BuiltinFunction::Primitive::Cosine:
		case HorseIR::BuiltinFunction::Primitive::Sine:
		case HorseIR::BuiltinFunction::Primitive::Tangent:
		case HorseIR::BuiltinFunction::Primitive::InverseCosine:
		case HorseIR::BuiltinFunction::Primitive::InverseSine:
		case HorseIR::BuiltinFunction::Primitive::InverseTangent:
		case HorseIR::BuiltinFunction::Primitive::HyperbolicCosine:
		case HorseIR::BuiltinFunction::Primitive::HyperbolicSine:
		case HorseIR::BuiltinFunction::Primitive::HyperbolicTangent:
		case HorseIR::BuiltinFunction::Primitive::HyperbolicInverseCosine:
		case HorseIR::BuiltinFunction::Primitive::HyperbolicInverseSine:
		case HorseIR::BuiltinFunction::Primitive::HyperbolicInverseTangent:
		
		// Date Unary
		case HorseIR::BuiltinFunction::Primitive::Date:
		case HorseIR::BuiltinFunction::Primitive::DateYear:
		case HorseIR::BuiltinFunction::Primitive::DateMonth:
		case HorseIR::BuiltinFunction::Primitive::DateDay:
		case HorseIR::BuiltinFunction::Primitive::Time:
		case HorseIR::BuiltinFunction::Primitive::TimeHour:
		case HorseIR::BuiltinFunction::Primitive::TimeMinute:
		case HorseIR::BuiltinFunction::Primitive::TimeSecond:
		case HorseIR::BuiltinFunction::Primitive::TimeMillisecond:

		// Binary
		case HorseIR::BuiltinFunction::Primitive::Less:
		case HorseIR::BuiltinFunction::Primitive::Greater:
		case HorseIR::BuiltinFunction::Primitive::LessEqual:
		case HorseIR::BuiltinFunction::Primitive::GreaterEqual:
		case HorseIR::BuiltinFunction::Primitive::Equal:
		case HorseIR::BuiltinFunction::Primitive::NotEqual:
		case HorseIR::BuiltinFunction::Primitive::Plus:
		case HorseIR::BuiltinFunction::Primitive::Minus:
		case HorseIR::BuiltinFunction::Primitive::Multiply:
		case HorseIR::BuiltinFunction::Primitive::Divide:
		case HorseIR::BuiltinFunction::Primitive::Power:
		case HorseIR::BuiltinFunction::Primitive::LogarithmBase:
		case HorseIR::BuiltinFunction::Primitive::Modulo:
		case HorseIR::BuiltinFunction::Primitive::And:
		case HorseIR::BuiltinFunction::Primitive::Or:
		case HorseIR::BuiltinFunction::Primitive::Nand:
		case HorseIR::BuiltinFunction::Primitive::Nor:
		case HorseIR::BuiltinFunction::Primitive::Xor:

		// Date Binary
		case HorseIR::BuiltinFunction::Primitive::DatetimeAdd:
		case HorseIR::BuiltinFunction::Primitive::DatetimeSubtract:
		case HorseIR::BuiltinFunction::Primitive::DatetimeDifference:

		// Algebraic Unary
		case HorseIR::BuiltinFunction::Primitive::Range:
		case HorseIR::BuiltinFunction::Primitive::Factorial:
		case HorseIR::BuiltinFunction::Primitive::Reverse:

		// Algebraic Binary
		case HorseIR::BuiltinFunction::Primitive::Random_k:
		case HorseIR::BuiltinFunction::Primitive::IndexOf:
		case HorseIR::BuiltinFunction::Primitive::Take:
		case HorseIR::BuiltinFunction::Primitive::Drop:
		case HorseIR::BuiltinFunction::Primitive::Vector:

		// Database
		case HorseIR::BuiltinFunction::Primitive::Keys:
		case HorseIR::BuiltinFunction::Primitive::Values:
		case HorseIR::BuiltinFunction::Primitive::Fetch:
		case HorseIR::BuiltinFunction::Primitive::JoinIndex:
		{
			return {true, Synchronization::None};
		}

		// Algebraic Binary
		case HorseIR::BuiltinFunction::Primitive::Member:
		{
			if (index == 1)
			{
				return {true, Synchronization::In};
			}
			return {true, Synchronization::None};
		}

		// Indexing
		case HorseIR::BuiltinFunction::Primitive::Index:
		{
			if (index == 0)
			{
				return {true, Synchronization::In};
			}
			return {true, Synchronization::None};

		}
		case HorseIR::BuiltinFunction::Primitive::IndexAssignment:
		{
			if (index == 0)
			{
				return {true, Synchronization::In | Synchronization::Out};
			}
			return {true, Synchronization::Out};
		}

		// --------------------
		// Compression Geometry
		// --------------------

		// Algebraic Unary
		case HorseIR::BuiltinFunction::Primitive::Where:

		// Algebraic Binary
		case HorseIR::BuiltinFunction::Primitive::Compress:
		{
			return {true, Synchronization::None};
		}

		// ----------------------
		// Independent Operations
		// ----------------------

		// Algebraic Unary
		case HorseIR::BuiltinFunction::Primitive::Unique:
		case HorseIR::BuiltinFunction::Primitive::Group:

		// Algebraic Binary
		case HorseIR::BuiltinFunction::Primitive::Order:
		{
			return {true, (Synchronization::In | Synchronization::Out)};
		}

		// --------------------
		// Reduction Operations
		// --------------------

		// Reduction
		case HorseIR::BuiltinFunction::Primitive::Length:
		case HorseIR::BuiltinFunction::Primitive::Sum:
		case HorseIR::BuiltinFunction::Primitive::Average:
		case HorseIR::BuiltinFunction::Primitive::Minimum:
		case HorseIR::BuiltinFunction::Primitive::Maximum:
		{
			return {true, (Synchronization::Out | Synchronization::Reduction)};
		}

		// ---------------
		// List Operations
		// ---------------

		// List
		case HorseIR::BuiltinFunction::Primitive::Each:
		case HorseIR::BuiltinFunction::Primitive::EachItem:
		case HorseIR::BuiltinFunction::Primitive::EachLeft:
		case HorseIR::BuiltinFunction::Primitive::EachRight:
		{
			const auto type = arguments.at(0)->GetType();
			const auto function = HorseIR::TypeUtils::GetType<HorseIR::FunctionType>(type)->GetFunctionDeclaration();
			return AnalyzeCall(function, {}, index - 1); // Nested function properties
		}

		// ---------------
		// Cell Operations
		// ---------------

		// List
		case HorseIR::BuiltinFunction::Primitive::Raze:
		{
			return {true, (Synchronization::Out | Synchronization::Raze)};
		}

		// --------------
		// CPU Operations
		// --------------

		// Algebraic Unary
		case HorseIR::BuiltinFunction::Primitive::Random:
		case HorseIR::BuiltinFunction::Primitive::Seed:
		case HorseIR::BuiltinFunction::Primitive::Flip:

		// Algebraic Binary
		case HorseIR::BuiltinFunction::Primitive::Append:
		case HorseIR::BuiltinFunction::Primitive::Replicate:
		case HorseIR::BuiltinFunction::Primitive::Like:

		// List
		case HorseIR::BuiltinFunction::Primitive::List:
		case HorseIR::BuiltinFunction::Primitive::ToList:
		case HorseIR::BuiltinFunction::Primitive::Match:

		// Database
		case HorseIR::BuiltinFunction::Primitive::Enum:
		case HorseIR::BuiltinFunction::Primitive::Dictionary:
		case HorseIR::BuiltinFunction::Primitive::Table:
		case HorseIR::BuiltinFunction::Primitive::KeyedTable:
		case HorseIR::BuiltinFunction::Primitive::Meta:
		case HorseIR::BuiltinFunction::Primitive::ColumnValue:
		case HorseIR::BuiltinFunction::Primitive::LoadTable:

		// Other
		case HorseIR::BuiltinFunction::Primitive::LoadCSV:
		case HorseIR::BuiltinFunction::Primitive::Print:
		case HorseIR::BuiltinFunction::Primitive::Format:
		case HorseIR::BuiltinFunction::Primitive::String:
		case HorseIR::BuiltinFunction::Primitive::SubString:
		{
			return {false, Synchronization::None};
		}
	}
	
	Utils::Logger::LogError("GPU analysis helper does not support builtin function '" + function->GetName() + "'");
}

void GPUAnalysisHelper::Visit(const HorseIR::CastExpression *cast)
{
	cast->GetExpression()->Accept(*this);
}

void GPUAnalysisHelper::Visit(const HorseIR::Literal *literal)
{
	m_gpu = false;
	m_synchronization = Synchronization::None;
}

void GPUAnalysisHelper::Visit(const HorseIR::Identifier *identifier)
{
	m_gpu = true;
	m_synchronization = Synchronization::None;
}

}
