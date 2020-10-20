#pragma once

#include "HorseIR/Traversal/ConstVisitor.h"

#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/TypeUtils.h"

#include "Utils/Logger.h"

namespace HorseIR {
namespace Analysis {

class ValueAnalysisHelper
{
public:
	static bool IsConstant(const Expression *expression)
	{
		ConstantHelper helper;
		helper.Analyze(expression);
		return helper.IsConstant();
	}

	template<typename T>
	static T GetScalar(const Expression *expression)
	{
		ValueHelper<T> helper;
		helper.Analyze(expression);
		return helper.GetValue().at(0);
	}

	template<typename T>
	static std::vector<T> GetConstant(const Expression *expression)
	{
		ValueHelper<T> helper;
		helper.Analyze(expression);
		return helper.GetValue();
	}

private:
	class ConstantHelper : public ConstVisitor
	{
	public:
		void Analyze(const Expression *expression)
		{
			m_constant = false;
			expression->Accept(*this);
		}

		void Visit(const Literal *expression) override
		{
			m_constant = true;
		}

		bool IsConstant() const { return m_constant; }

	private:
		bool m_constant = false;
	};

	template<typename T>
	class ValueHelper : public ConstVisitor
	{
	public:
		void Analyze(const Expression *expression)
		{
			m_value.clear();
			expression->Accept(*this);
		}                               

		void Visit(const CallExpression *expression) override
		{
			Utils::Logger::LogError("Unexpected call expression when fetching constant.");
		}

		void Visit(const CastExpression *expression) override
		{
			Utils::Logger::LogError("Unexpected cast expression when fetching constant.");
		}

		template<typename S>
		void VisitLiteral(const S *literal)
		{
			if constexpr(std::is_same<typename S::Type, T>::value)
			{
				m_value = literal->GetValues();
			}
			else if constexpr(std::is_convertible<typename S::Type, T>::value)
			{
				for (const auto& value : literal->GetValues())
				{
					m_value.push_back(value);
				}
			}
			else
			{
				Utils::Logger::LogError("Unexpected constant of type " + TypeUtils::TypeString(literal->GetTypes()));
			}
		}

		void Visit(const BooleanLiteral *literal) override
		{
			VisitLiteral(literal);
		}

		void Visit(const CharLiteral *literal) override
		{
			VisitLiteral(literal);
		}

		void Visit(const Int8Literal *literal) override
		{
			VisitLiteral(literal);
		}

		void Visit(const Int16Literal *literal) override
		{
			VisitLiteral(literal);
		}

		void Visit(const Int32Literal *literal) override
		{
			VisitLiteral(literal);
		}

		void Visit(const Int64Literal *literal) override
		{
			VisitLiteral(literal);
		}

		void Visit(const Float32Literal *literal) override
		{
			VisitLiteral(literal);
		}

		void Visit(const Float64Literal *literal) override
		{
			VisitLiteral(literal);
		}

		void Visit(const ComplexLiteral *literal) override
		{
			VisitLiteral(literal);
		}

		void Visit(const SymbolLiteral *literal) override
		{
			VisitLiteral(literal);
		}

		void Visit(const StringLiteral *literal) override
		{
			VisitLiteral(literal);
		}

		void Visit(const DatetimeLiteral *literal) override
		{
			VisitLiteral(literal);
		}

		void Visit(const DateLiteral *literal) override
		{
			VisitLiteral(literal);
		}

		void Visit(const MonthLiteral *literal) override
		{
			VisitLiteral(literal);
		}

		void Visit(const MinuteLiteral *literal) override
		{
			VisitLiteral(literal);
		}

		void Visit(const SecondLiteral *literal) override
		{
			VisitLiteral(literal);
		}

		void Visit(const TimeLiteral *literal) override
		{
			VisitLiteral(literal);
		}

		const std::vector<T>& GetValue() const { return m_value; }
	
	private:
		std::vector<T> m_value;
	};
};

}
}
