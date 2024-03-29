#pragma once

#include "PTX/Tree/Node.h"

namespace PTX {

class Declaration : public Node
{
public:
	enum class LinkDirective {
		None,
		External,
		Visible,
		Common,
		Weak
	};

	static std::string LinkDirectiveString(LinkDirective linkDirective)
	{
		switch (linkDirective)
		{
			case LinkDirective::None:
				return "";
			case LinkDirective::External:
				return ".extern";
			case LinkDirective::Visible:
				return ".visible";
			case LinkDirective::Common:
				return ".common";
			case LinkDirective::Weak:
				return ".weak";
		}
		return ".<unknown>";
	}

	Declaration(LinkDirective linkDirective = LinkDirective::None) : m_linkDirective(linkDirective) {}

	// Properties

	LinkDirective GetLinkDirective() const { return m_linkDirective; }
	void SetLinkDirective(LinkDirective linkDirective) { m_linkDirective = linkDirective; }

protected:
	LinkDirective m_linkDirective = LinkDirective::None;
};

}
