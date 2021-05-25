#include "HorseIR/Analysis/Helpers/GPUAnalysisHelper.h"

#include "HorseIR/Utils/TypeUtils.h"

#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace HorseIR {
namespace Analysis {

GPUAnalysisHelper::Device GPUAnalysisHelper::IsGPU(const Statement *statement)
{
	// Reset the analysis and traverse

	m_device = Device::CPU;
	m_synchronization = Synchronization::None;
	statement->Accept(*this);

	return m_device;
}

bool GPUAnalysisHelper::IsSynchronized(const Statement *source, const Statement *destination, unsigned int index)
{
	// Check for synchronization between the two statements. Non-GPU links are never marked as synchronized

	// Check for synchronization on the source statement

	m_device = Device::CPU;
	m_synchronization = Synchronization::None;

	source->Accept(*this);
	if (m_device == Device::CPU || m_device == Device::GPULibrary)
	{
		return false;
	}

	auto sourceSynchronization = m_synchronization;

	// Check for synchronization on the destination statement at the index

	m_index = index;
	m_device = Device::CPU;
	m_synchronization = Synchronization::None;

	destination->Accept(*this);
	if (m_device == Device::CPU || m_device == Device::GPULibrary)
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

void GPUAnalysisHelper::Visit(const DeclarationStatement *declarationS)
{
	m_device = Device::GPU;
	m_synchronization = Synchronization::None;
}

void GPUAnalysisHelper::Visit(const AssignStatement *assignS)
{
	assignS->GetExpression()->Accept(*this);
}

void GPUAnalysisHelper::Visit(const ExpressionStatement *expressionS)
{
	expressionS->GetExpression()->Accept(*this);
}

void GPUAnalysisHelper::Visit(const ReturnStatement *returnS)
{
	// Explicitly disallow return from kernels, we will insert as needed

	m_device = Device::CPU;
	m_synchronization = Synchronization::None;
}

void GPUAnalysisHelper::Visit(const CallExpression *call)
{
	auto [device, synchronization] = AnalyzeCall(call->GetFunctionLiteral()->GetFunction(), call->GetArguments(), m_index);

	m_device = device;
	m_synchronization = synchronization;
}

std::pair<GPUAnalysisHelper::Device, GPUAnalysisHelper::Synchronization> GPUAnalysisHelper::AnalyzeCall(const FunctionDeclaration *function, const std::vector<const Operand *>& arguments, unsigned int index)
{
	switch (function->GetKind())
	{
		case FunctionDeclaration::Kind::Builtin:
			return AnalyzeCall(static_cast<const BuiltinFunction *>(function), arguments, index);
		case FunctionDeclaration::Kind::Definition:
			return AnalyzeCall(static_cast<const Function *>(function), arguments, index);
		default:
			Utils::Logger::LogError("Unsupported function kind");
	}
}

std::pair<GPUAnalysisHelper::Device, GPUAnalysisHelper::Synchronization> GPUAnalysisHelper::AnalyzeCall(const Function *function, const std::vector<const Operand *>& arguments, unsigned int index)
{
	return std::make_pair(Device::CPU, Synchronization::None);
}

std::pair<GPUAnalysisHelper::Device, GPUAnalysisHelper::Synchronization> GPUAnalysisHelper::AnalyzeCall(const BuiltinFunction *function, const std::vector<const Operand *>& arguments, unsigned int index)
{
	switch (function->GetPrimitive())
	{
		// ---------------
		// Vector Geometry
		// ---------------

		// Unary
		case BuiltinFunction::Primitive::Absolute:
		case BuiltinFunction::Primitive::Negate:
		case BuiltinFunction::Primitive::Ceiling:
		case BuiltinFunction::Primitive::Floor:
		case BuiltinFunction::Primitive::Round:
		case BuiltinFunction::Primitive::Conjugate:
		case BuiltinFunction::Primitive::Reciprocal:
		case BuiltinFunction::Primitive::Sign:
		case BuiltinFunction::Primitive::Pi:
		case BuiltinFunction::Primitive::Not:
		case BuiltinFunction::Primitive::Logarithm:
		case BuiltinFunction::Primitive::Logarithm2:
		case BuiltinFunction::Primitive::Logarithm10:
		case BuiltinFunction::Primitive::SquareRoot:
		case BuiltinFunction::Primitive::Exponential:
		case BuiltinFunction::Primitive::Cosine:
		case BuiltinFunction::Primitive::Sine:
		case BuiltinFunction::Primitive::Tangent:
		case BuiltinFunction::Primitive::InverseCosine:
		case BuiltinFunction::Primitive::InverseSine:
		case BuiltinFunction::Primitive::InverseTangent:
		case BuiltinFunction::Primitive::HyperbolicCosine:
		case BuiltinFunction::Primitive::HyperbolicSine:
		case BuiltinFunction::Primitive::HyperbolicTangent:
		case BuiltinFunction::Primitive::HyperbolicInverseCosine:
		case BuiltinFunction::Primitive::HyperbolicInverseSine:
		case BuiltinFunction::Primitive::HyperbolicInverseTangent:
		
		// Date Unary
		case BuiltinFunction::Primitive::Date:
		case BuiltinFunction::Primitive::DateYear:
		case BuiltinFunction::Primitive::DateMonth:
		case BuiltinFunction::Primitive::DateDay:
		case BuiltinFunction::Primitive::Time:
		case BuiltinFunction::Primitive::TimeHour:
		case BuiltinFunction::Primitive::TimeMinute:
		case BuiltinFunction::Primitive::TimeSecond:
		case BuiltinFunction::Primitive::TimeMillisecond:

		// Binary
		case BuiltinFunction::Primitive::Equal:
		case BuiltinFunction::Primitive::NotEqual:
		case BuiltinFunction::Primitive::Plus:
		case BuiltinFunction::Primitive::Minus:
		case BuiltinFunction::Primitive::Multiply:
		case BuiltinFunction::Primitive::Divide:
		case BuiltinFunction::Primitive::Power:
		case BuiltinFunction::Primitive::LogarithmBase:
		case BuiltinFunction::Primitive::Modulo:
		case BuiltinFunction::Primitive::And:
		case BuiltinFunction::Primitive::Or:
		case BuiltinFunction::Primitive::Nand:
		case BuiltinFunction::Primitive::Nor:
		case BuiltinFunction::Primitive::Xor:

		// Date Binary
		case BuiltinFunction::Primitive::DatetimeAdd:
		case BuiltinFunction::Primitive::DatetimeSubtract:
		case BuiltinFunction::Primitive::DatetimeDifference:

		// Algebraic Unary
		case BuiltinFunction::Primitive::Range:
		case BuiltinFunction::Primitive::Factorial:

		// Algebraic Binary
		case BuiltinFunction::Primitive::Random_k:
		case BuiltinFunction::Primitive::Take:
		case BuiltinFunction::Primitive::Drop:
		case BuiltinFunction::Primitive::Vector:
		{
			return std::make_pair(Device::GPU, Synchronization::None);
		}

		case BuiltinFunction::Primitive::Reverse:
		{
			// Indexed write
			return std::make_pair(Device::GPU, Synchronization::In);
		}

		// Binary
		case BuiltinFunction::Primitive::Less:
		case BuiltinFunction::Primitive::Greater:
		case BuiltinFunction::Primitive::LessEqual:
		case BuiltinFunction::Primitive::GreaterEqual:
		{
			const auto type0 = arguments.at(0)->GetType();
			if (TypeUtils::IsCharacterType(type0))
			{
				return std::make_pair(Device::CPU, Synchronization::None);
			}
			else if (const auto listType0 = TypeUtils::GetType<ListType>(type0))
			{
				if (TypeUtils::ForanyElement(listType0, TypeUtils::IsCharacterType))
				{
					return std::make_pair(Device::CPU, Synchronization::None);
				}
			}

			if (arguments.size() == 2)
			{
				const auto type1 = arguments.at(1)->GetType();
				if (const auto listType1 = TypeUtils::GetType<ListType>(type1))
				{
					if (TypeUtils::ForanyElement(listType1, TypeUtils::IsCharacterType))
					{
						return std::make_pair(Device::CPU, Synchronization::None);
					}
				}
				else if (TypeUtils::IsCharacterType(type1))
				{
					return std::make_pair(Device::CPU, Synchronization::None);
				}
			}
			return std::make_pair(Device::GPU, Synchronization::None);
		}

		// Algebraic Binary
		case BuiltinFunction::Primitive::IndexOf:
		case BuiltinFunction::Primitive::Member:
		{
			if (index == 1)
			{
				return std::make_pair(Device::GPU, Synchronization::In);
			}
			return std::make_pair(Device::GPU, Synchronization::None);
		}

		// Indexing
		case BuiltinFunction::Primitive::Index:
		{
			const auto type0 = arguments.at(0)->GetType();
			if (TypeUtils::IsType<ListType>(type0))
			{
				return std::make_pair(Device::CPU, Synchronization::None);
			}

			if (index == 0)
			{
				return std::make_pair(Device::GPU, Synchronization::In);
			}
			return std::make_pair(Device::GPU, Synchronization::None);

		}
		case BuiltinFunction::Primitive::IndexAssignment:
		{
			if (index == 0)
			{
				return std::make_pair(Device::GPU, Synchronization::In | Synchronization::Out);
			}
			return std::make_pair(Device::GPU, Synchronization::Out);
		}

		// Algebraic Binary
		case BuiltinFunction::Primitive::Append:
		{
			const auto type0 = arguments.at(0)->GetType();
			if (!TypeUtils::IsType<BasicType>(type0))
			{
				return std::make_pair(Device::CPU, Synchronization::None);
			}

			return std::make_pair(Device::GPU, Synchronization::In);
		}

		// --------------------
		// Compression Geometry
		// --------------------

		// Algebraic Unary
		case BuiltinFunction::Primitive::Where:

		// Algebraic Binary
		case BuiltinFunction::Primitive::Compress:
		{
			return std::make_pair(Device::GPU, Synchronization::None);
		}

		// ----------------------
		// Independent Operations
		// ----------------------

		case BuiltinFunction::Primitive::Unique:
		{
			switch (Utils::Options::GetAlgorithm_UniqueKind())
			{
				case Utils::Options::UniqueKind::SortUnique:
					return std::make_pair(Device::GPULibrary, Synchronization::None);
				case Utils::Options::UniqueKind::LoopUnique:
					return std::make_pair(Device::GPU, Synchronization::In);
			}
		}

		// Algebraic Binary
		case BuiltinFunction::Primitive::Order:
		{
			auto type0 = arguments.at(0)->GetType();
			if (const auto listType0 = TypeUtils::GetType<ListType>(type0))
			{
				if (TypeUtils::ForanyElement(listType0, TypeUtils::IsCharacterType))
				{
					return std::make_pair(Device::CPU, Synchronization::None);
				}
			}
			else if (TypeUtils::IsCharacterType(type0))
			{
				return std::make_pair(Device::CPU, Synchronization::None);
			}
		}

		// Algebraic Unary
		case BuiltinFunction::Primitive::Group:

		// Database
		case BuiltinFunction::Primitive::JoinIndex:
		{
			// Complex independent operations are controlled on the CPU with GPU sections

			return std::make_pair(Device::GPULibrary, Synchronization::None);
		}

		// --------------------
		// Reduction Operations
		// --------------------

		// Reduction
		case BuiltinFunction::Primitive::Length:
		case BuiltinFunction::Primitive::Sum:
		case BuiltinFunction::Primitive::Average:
		case BuiltinFunction::Primitive::Minimum:
		case BuiltinFunction::Primitive::Maximum:
		{
			return std::make_pair(Device::GPU, (Synchronization::Out | Synchronization::Reduction));
		}

		// ---------------
		// List Operations
		// ---------------

		// List
		case BuiltinFunction::Primitive::Each:
		case BuiltinFunction::Primitive::EachItem:
		case BuiltinFunction::Primitive::EachLeft:
		case BuiltinFunction::Primitive::EachRight:
		{
			const auto type = arguments.at(0)->GetType();
			const auto function = TypeUtils::GetType<FunctionType>(type)->GetFunctionDeclaration();

			std::vector<const Operand *> nestedArguments(std::begin(arguments) + 1, std::end(arguments));
			return AnalyzeCall(function, nestedArguments, index - 1); // Nested function properties
		}

		// ---------------
		// Cell Operations
		// ---------------

		// List
		case BuiltinFunction::Primitive::Raze:
		{
			return std::make_pair(Device::GPU, (Synchronization::Out | Synchronization::Raze));
		}
		case BuiltinFunction::Primitive::ToList:
		{
			return std::make_pair(Device::GPU, Synchronization::In);
		}

		// ----------------------
		// GPU Library Operations
		// ----------------------

		case BuiltinFunction::Primitive::GPUOrderLib:
		case BuiltinFunction::Primitive::GPUGroupLib:
		case BuiltinFunction::Primitive::GPUUniqueLib:
		case BuiltinFunction::Primitive::GPULoopJoinLib:
		case BuiltinFunction::Primitive::GPUHashJoinLib:
		{
			return std::make_pair(Device::CPU, Synchronization::None);
		}
		case BuiltinFunction::Primitive::GPUOrderInit:
		case BuiltinFunction::Primitive::GPUOrder:
		case BuiltinFunction::Primitive::GPUOrderShared:

		case BuiltinFunction::Primitive::GPUGroup:

		case BuiltinFunction::Primitive::GPUUnique:

		case BuiltinFunction::Primitive::GPULoopJoinCount:
		case BuiltinFunction::Primitive::GPULoopJoin:

		case BuiltinFunction::Primitive::GPUHashCreate:
		case BuiltinFunction::Primitive::GPUHashJoinCount:
		case BuiltinFunction::Primitive::GPUHashJoin:
		{
			return std::make_pair(Device::GPU, (Synchronization::In | Synchronization::Out));
		}

		// --------------
		// CPU Operations
		// --------------

		// Algebraic Unary
		case BuiltinFunction::Primitive::Random:
		case BuiltinFunction::Primitive::Seed:
		case BuiltinFunction::Primitive::Flip:

		// Algebraic Binary
		case BuiltinFunction::Primitive::Replicate:
		case BuiltinFunction::Primitive::Like:

		// List
		case BuiltinFunction::Primitive::List:
		case BuiltinFunction::Primitive::Match:

		// Database
		case BuiltinFunction::Primitive::Enum:
		case BuiltinFunction::Primitive::Dictionary:
		case BuiltinFunction::Primitive::Table:
		case BuiltinFunction::Primitive::KeyedTable:
		case BuiltinFunction::Primitive::Keys:
		case BuiltinFunction::Primitive::Values:
		case BuiltinFunction::Primitive::Meta:
		case BuiltinFunction::Primitive::Fetch:
		case BuiltinFunction::Primitive::ColumnValue:
		case BuiltinFunction::Primitive::LoadTable:

		// Other
		case BuiltinFunction::Primitive::LoadCSV:
		case BuiltinFunction::Primitive::Print:
		case BuiltinFunction::Primitive::String:
		case BuiltinFunction::Primitive::SubString:
		{
			return std::make_pair(Device::CPU, Synchronization::None);
		}
	}
	
	Utils::Logger::LogError("GPU analysis helper does not support builtin function '" + function->GetName() + "'");
}

void GPUAnalysisHelper::Visit(const CastExpression *cast)
{
	cast->GetExpression()->Accept(*this);
}

void GPUAnalysisHelper::Visit(const Literal *literal)
{
	m_device = Device::CPU;
	m_synchronization = Synchronization::None;
}

void GPUAnalysisHelper::Visit(const Identifier *identifier)
{
	m_device = Device::GPU;
	m_synchronization = Synchronization::None;
}

}
}
