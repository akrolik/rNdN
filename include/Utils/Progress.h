#include <string>

namespace Utils {

class Progress
{
public:
	static Progress Start(const std::string& name, bool print);

	// Status

	void Update(unsigned int complete, unsigned int total, bool approx = false);
	void Complete();

	// Width

	static unsigned int GetWidth() { return s_width; }
	static void SetWidth(unsigned int width) { s_width = width; }

private:
	Progress(const std::string& name, bool print) : m_name(name), m_print(print) {}

	std::string m_name;
	bool m_print = false;
	static unsigned int s_width;
};

}
