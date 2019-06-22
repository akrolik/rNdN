#pragma once

#include "HorseIR/Traversal/ConstVisitor.h"

#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/TypeUtils.h"

#include "Utils/Logger.h"

namespace Analysis {

class ValueAnalysisHelper
{
public:
	static bool IsConstant(const HorseIR::Expression *expression)
	{
		ConstantHelper helper;
		helper.Analyze(expression);
		return helper.IsConstant();
	}

	template<typename T>
	static std::vector<T> GetConstant(const HorseIR::Expression *expression)
	{
		ValueHelper<T> helper;
		helper.Analyze(expression);
		return helper.GetValue();
	}

private:
	class ConstantHelper : public HorseIR::ConstVisitor
	{
	public:
		void Analyze(const HorseIR::Expression *expression)
		{
			m_constant = false;
			expression->Accept(*this);
		}

		void Visit(const HorseIR::Literal *expression) override
		{
			m_constant = true;
		}

		bool IsConstant() const { return m_constant; }

	private:
		bool m_constant = false;
	};

	template<typename T>
	class ValueHelper : public HorseIR::ConstVisitor
	{
	public:
		void Analyze(const HorseIR::Expression *expression)
		{
			m_value.clear();
			expression->Accept(*this);
		}                               

		void Visit(const HorseIR::CallExpression *expression) override
		{
			Utils::Logger::LogError("Unexpected call expression when fetching constant.");
		}

		void Visit(const HorseIR::CastExpression *expression) override
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
				Utils::Logger::LogError("Unexpected constant of type " + HorseIR::TypeUtils::TypeString(literal->GetTypes()));
			}
		}

		void Visit(const HorseIR::BooleanLiteral *literal) override
		{
			VisitLiteral(literal);
		}

		void Visit(const HorseIR::CharLiteral *literal) override
		{
			VisitLiteral(literal);
		}

		void Visit(const HorseIR::Int8Literal *literal) override
		{
			VisitLiteral(literal);
		}

		void Visit(const HorseIR::Int16Literal *literal) override
		{
			VisitLiteral(literal);
		}

		void Visit(const HorseIR::Int32Literal *literal) override
		{
			VisitLiteral(literal);
		}

		void Visit(const HorseIR::Int64Literal *literal) override
		{
			VisitLiteral(literal);
		}

		void Visit(const HorseIR::Float32Literal *literal) override
		{
			VisitLiteral(literal);
		}

		void Visit(const HorseIR::Float64Literal *literal) override
		{
			VisitLiteral(literal);
		}

		void Visit(const HorseIR::ComplexLiteral *literal) override
		{
			VisitLiteral(literal);
		}

		void Visit(const HorseIR::SymbolLiteral *literal) override
		{
			VisitLiteral(literal);
		}

		void Visit(const HorseIR::StringLiteral *literal) override
		{
			VisitLiteral(literal);
		}

		void Visit(const HorseIR::DatetimeLiteral *literal) override
		{
			VisitLiteral(literal);
		}

		void Visit(const HorseIR::DateLiteral *literal) override
		{
			VisitLiteral(literal);
		}

		void Visit(const HorseIR::MonthLiteral *literal) override
		{
			VisitLiteral(literal);
		}

		void Visit(const HorseIR::MinuteLiteral *literal) override
		{
			VisitLiteral(literal);
		}

		void Visit(const HorseIR::SecondLiteral *literal) override
		{
			VisitLiteral(literal);
		}

		void Visit(const HorseIR::TimeLiteral *literal) override
		{
			VisitLiteral(literal);
		}

		const std::vector<T>& GetValue() const { return m_value; }
	
	private:
		std::vector<T> m_value;
	};
};

}
