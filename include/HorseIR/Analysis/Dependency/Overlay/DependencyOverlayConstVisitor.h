#pragma once

namespace HorseIR {
namespace Analysis {

class DependencyOverlay;

class FunctionDependencyOverlay;
class IfDependencyOverlay;
class WhileDependencyOverlay;
class RepeatDependencyOverlay;

class DependencyOverlayConstVisitor
{
public:
	virtual void Visit(const DependencyOverlay *node);

	virtual void Visit(const FunctionDependencyOverlay *node);
	virtual void Visit(const IfDependencyOverlay *node);
	virtual void Visit(const WhileDependencyOverlay *node);
	virtual void Visit(const RepeatDependencyOverlay *node);
};

}
}
