#pragma once

#include <string>

#include "PTX/Declarations/Declaration.h"

#include "PTX/Block.h"

namespace PTX {

class Function : public Declaration
{
public:
	void SetName(std::string name) { m_name = name; }
	std::string GetName() const { return m_name; }

	void SetBody(Block *body) { m_body = body; }

protected:
	std::string m_name;
	Block *m_body = nullptr;
};

}
