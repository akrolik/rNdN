#pragma once

#include <map>
#include <string>
#include <vector>

#include "PTX/Block.h"
#include "PTX/ParameterSpace.h"
#include "PTX/Type.h"

namespace PTX {

class Function
{
public:
	bool IsVisible() { return m_visible; }
	void SetVisible(bool visible) { m_visible = visible; }

	bool IsEntry() { return m_entry; }
	void SetEntry(bool entry) { m_entry = entry; }

	void SetName(std::string name) { m_name = name; }
	std::string GetName() { return m_name; }

	void SetReturnSpace(StateSpace *returnSpace) { m_returnSpace = returnSpace; }
	void AddParameter(ParameterSpace *parameterSpace) { m_parameterSpaces.push_back(parameterSpace); }
	void SetBody(Block *body) { m_body = body; }

	std::string ToString();

private:
	bool m_visible = false;
	bool m_entry = false;
	std::string m_name;

	std::vector<ParameterSpace *> m_parameterSpaces;
	StateSpace *m_returnSpace;
	Block *m_body;
};

}
