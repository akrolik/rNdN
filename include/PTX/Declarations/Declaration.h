#pragma once

namespace PTX {

class Declaration
{
public:
	enum LinkDirective {
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
			case None:
				return "";
			case Extern:
				return ".extern";
			case Visible:
				return ".visible";
			case Common:
				return ".common";
			case Weak:
				return ".weak";
		}
		return ".<unknown>";
	}

	Declaration(LinkDirective linkDirective = None) : m_linkDirective(linkDirective) {}

	LinkDirective GetLinkDirective() const { return m_linkDirective; }
	void SetLinkDirective(LinkDirective linkDirective) { m_linkDirective = linkDirective; }

	virtual std::string ToString() const = 0;

protected:
	LinkDirective m_linkDirective = None;
};

}
