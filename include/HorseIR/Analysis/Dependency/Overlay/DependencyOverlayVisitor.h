#pragma once

namespace HorseIR {
namespace Analysis {

class DependencyOverlay;

class FunctionDependencyOverlay;
class IfDependencyOverlay;
class WhileDependencyOverlay;
class RepeatDependencyOverlay;

class DependencyOverlayVisitor
{
public:
	virtual void Visit(DependencyOverlay *node);

	virtual void Visit(FunctionDependencyOverlay *node);
	virtual void Visit(IfDependencyOverlay *node);
	virtual void Visit(WhileDependencyOverlay *node);
	virtual void Visit(RepeatDependencyOverlay *node);
};

}
}
