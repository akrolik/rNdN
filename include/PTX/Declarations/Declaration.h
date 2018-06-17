#pragma once

namespace PTX {

class Declaration
{
public:
	enum class LinkDirective {
		None,
		Extern,
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
			case LinkDirective::Extern:
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

	LinkDirective GetLinkDirective() const { return m_linkDirective; }
	void SetLinkDirective(LinkDirective linkDirective) { m_linkDirective = linkDirective; }

	virtual std::string ToString() const = 0;

protected:
	LinkDirective m_linkDirective = LinkDirective::None;
};

}
