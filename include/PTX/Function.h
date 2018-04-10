#pragma once

#include <string>

#include "PTX/Block.h"

namespace PTX {

class Function
{
public:
	bool IsVisible() { return m_visible; }
	void SetVisible(bool visible) { m_visible = visible; }

	void SetName(std::string name) { m_name = name; }
	std::string GetName() { return m_name; }

	void SetBody(Block *body) { m_body = body; }

	virtual std::string ToString() = 0;

protected:
	bool m_visible = false;
	std::string m_name;
	Block *m_body = nullptr;
};

}
