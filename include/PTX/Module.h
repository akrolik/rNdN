#pragma once

#include <string>
#include <sstream>
#include <vector>

#include "PTX/Declarations/Declaration.h"
#include "PTX/Type.h"

#include "Libraries/json.hpp"

namespace PTX {

class Module
{
public:
	void SetVersion(unsigned int major, unsigned int minor) { m_versionMajor = major; m_versionMinor = minor; }
	void SetDeviceTarget(const std::string& target) { m_target = target; }
	void SetAddressSize(Bits addressSize) { m_addressSize = addressSize; }

	void AddDeclaration(Declaration *declaration)
	{
		m_declarations.push_back(declaration);
	}

	std::string ToString() const
	{
		std::ostringstream code;

		code << ".version " << m_versionMajor << "." << m_versionMinor << std::endl;
		code << ".target " << m_target << std::endl;
		code << ".address_size " << std::to_string(static_cast<int>(m_addressSize)) << std::endl;

		for (const auto& declaration : m_declarations)
		{
			code << declaration->ToString();
		}

		return code.str();
	}

	json ToJSON() const
	{
		json j;
		j["version_major"] = m_versionMajor;
		j["version_minor"] = m_versionMinor;
		j["target"] = m_target;
		j["address_size"] = static_cast<int>(m_addressSize);
		for (const auto& declaration : m_declarations)
		{
			j["declarations"].push_back(declaration->ToJSON());
		}
		return j;
	}

private:
	unsigned int m_versionMajor, m_versionMinor;
	std::string m_target;
	Bits m_addressSize = Bits::Bits32;

	std::vector<Declaration *> m_declarations;
};

}
