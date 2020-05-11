#include <iostream>
#include <string>

namespace Utils {

class Progress
{
public:
	static Progress Start(const std::string& name)
	{
		Progress p(name);
		p.Update(0, 0);
		return p;
	}

	void Update(unsigned int complete, unsigned int total)
	{
		if (complete % 100 == 0 || complete == total)
		{
			auto progress = (total > 0) ? float(complete) / total : 0;

			std::cout << "[INFO] " << m_name << ": [";
			for (auto i = 0u; i < s_width; ++i)
			{
				if (i < s_width * progress)
				{
					std::cout << "#";
				}
				else
				{
					std::cout << " ";
				}
			}
			std::cout << "] ";
			if (total > 0)
			{
				std::cout << complete << "/" << total << " ";
			}
			std::cout << std::fixed << std::setprecision(1) << "(" << progress * 100 << " %)\r" << std::flush;
		}
	}

	void Complete()
	{
		std::cout << std::endl;
	}

	static unsigned int GetWidth() { return s_width; }
	static void SetWidth(unsigned int width) { s_width = width; }

private:
	Progress(const std::string& name) : m_name(name) {}

	std::string m_name;
	static unsigned int s_width;
};

unsigned int Progress::s_width = 30;

}
