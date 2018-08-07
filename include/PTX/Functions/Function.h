#pragma once

#include <string>

#include "PTX/Declarations/Declaration.h"

#include "PTX/Statements/StatementList.h"
#include "PTX/Statements/Statement.h"

#include "Libraries/json.hpp"

#include "Utils/Logger.h"

namespace PTX {

class Function : public Declaration
{
public:
	Function() : m_options(*this) {}
	Function(const std::string& name, Declaration::LinkDirective linkDirective = Declaration::LinkDirective::None) : Declaration(linkDirective), m_name(name), m_options(*this) {};

	void SetName(const std::string& name) { m_name = name; }
	const std::string& GetName() const { return m_name; }

	struct Options
	{
	public:
		Options(Function& function) : m_function(function) {}

		constexpr static unsigned int DynamicThreadCount = 0;

		unsigned int GetThreadCount() const { return m_threadCount; }
		void SetThreadCount(unsigned int count)
		{
			if (m_threadCount == count)
			{
				return;
			}

			// Check that the new configuration is compatible with the existing settings

			if (m_threadCount != DynamicThreadCount)
			{
				Utils::Logger::LogError("Thread count " + (count == DynamicThreadCount) ? "<dynamic>" : std::to_string(count) + " incompatible with thread count " + std::to_string(m_threadCount) + " in function " + m_function.m_name);
			}
			else if (m_threadMultiple != 0 && (count % m_threadMultiple) != 0)
			{
				Utils::Logger::LogError("Thread count " + (count == DynamicThreadCount) ? "<dynamic>" : std::to_string(count) + " incompatible with thread multiple " + std::to_string(m_threadMultiple) + " in function " + m_function.m_name);
			}

			m_threadCount = count;
		}

		unsigned int GetThreadMultiple() const { return m_threadMultiple; }
		void SetThreadMultiple(unsigned int multiple)
		{
			if (m_threadMultiple == multiple)
			{
				return;
			}

			if (m_threadMultiple == 0)
			{
				m_threadMultiple = multiple;
				return;
			}

			// Check that the new configuration is compatible with the existing settings
			// We choose the maximum multiple between the old/new provided that the larger
			// is a multiple of the smaller 

			if (multiple > m_threadMultiple && (multiple % m_threadMultiple) == 0)
			{
				if (m_threadCount != DynamicThreadCount && (m_threadCount % multiple) != 0)
				{
					Utils::Logger::LogError("Thread multiple " + std::to_string(multiple) + " incompatible with thread count " + std::to_string(m_threadCount) + " in function " + m_function.m_name);
				}
				m_threadMultiple = multiple;
			}
			else if (multiple < m_threadMultiple && (m_threadMultiple % multiple) == 0)
			{
				return;
			}
			else
			{
				Utils::Logger::LogError("Thread multiple " + std::to_string(multiple) + " incompatible with thread multiple " + std::to_string(m_threadMultiple) + " in function " + m_function.m_name);
			}
		}

		unsigned int GetSharedMemorySize() const { return m_sharedMemorySize; }
		void SetSharedMemorySize(unsigned int sharedMemorySize) { m_sharedMemorySize  = sharedMemorySize; }

		bool IsAtomicReturn() const { return m_atomicReturn; }
		void SetAtomicReturn(bool atomicReturn) { m_atomicReturn = atomicReturn; }

		std::string ToString() const
		{
			std::string output;
			output += "Thread count: " + ((m_threadCount == DynamicThreadCount) ? "<dynamic>" : std::to_string(m_threadCount)) + "\n";
			output += "Thread multiple: " + std::to_string(m_threadMultiple) + "\n";
			output += "Shared memory size: " + std::to_string(m_sharedMemorySize) + " bytes\n";
			output += "Atomic return: " + ((m_atomicReturn) ? std::string("true") : std::string("false"));
			return output;
		}

		json ToJSON() const
		{
			json j;
			j["thread_count"] = (m_threadCount == DynamicThreadCount) ? "<dynamic>" : std::to_string(m_threadCount);
			j["thread_multiple"] = std::to_string(m_threadMultiple);
			j["shared_memory"] = m_sharedMemorySize;
			j["atomic_return"] = m_atomicReturn;
			return j;
		}

	private:
		Function& m_function;

		unsigned int m_threadCount = DynamicThreadCount;
		unsigned int m_threadMultiple = 0;
		unsigned int m_sharedMemorySize = 0;

		bool m_atomicReturn = false;
	};

	Options& GetOptions() { return m_options; }
	const Options& GetOptions() const { return m_options; }

	std::string ToString(unsigned int indentation = 0) const override
	{
		std::string code;
		if (m_linkDirective != LinkDirective::None)
		{
			code += LinkDirectiveString(m_linkDirective) + " ";
		}
		code += GetDirectives() + " ";

		std::string ret = GetReturnString();
		if (ret.length() > 0)
		{
			code += "(" + ret + ") ";
		}
		code += m_name + "(" + GetParametersString() + ")";
		return code;
	}

	json ToJSON() const override
	{
		json j;
		j["kind"] = "PTX::Function";
		j["name"] = m_name;
		j["directives"] = GetDirectives();
		j["return"] = json::object();
		j["parameters"] = json::array();
		j["options"] = m_options.ToJSON();
		return j;
	}

private:
	virtual std::string GetDirectives() const = 0;
	virtual std::string GetReturnString() const = 0;
	virtual std::string GetParametersString() const = 0;

	std::string m_name;

	Options m_options;
};

}
