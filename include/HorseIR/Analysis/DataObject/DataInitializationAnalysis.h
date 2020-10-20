#pragma once

#include <unordered_map>
#include <utility>

#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"

#include "HorseIR/Analysis/DataObject/DataObjectAnalysis.h"

#include "HorseIR/Tree/Tree.h"

namespace HorseIR {
namespace Analysis {

class DataInitializationAnalysis : public ConstHierarchicalVisitor
{
public:
	DataInitializationAnalysis(const DataObjectAnalysis& objectAnalysis) : m_objectAnalysis(objectAnalysis) {}

	// Analysis input/output

	void Analyze(const Function *function);

	enum class Initialization {
		Clear,
		Minimum,
		Maximum,
		Copy,
		None
	};

	Initialization GetInitialization(const DataObject *returnObject) const
	{
		if (m_dataInit.find(returnObject) == m_dataInit.end())
		{
			return Initialization::None;
		}
		return m_dataInit.at(returnObject);
	}
	std::unordered_map<const DataObject *, Initialization> GetDataInitializations() const { return m_dataInit; }

	bool ContainsDataCopy(const DataObject *returnObject) const
	{
		return (m_dataCopies.find(returnObject) != m_dataCopies.end());
	}
	const DataObject *GetDataCopy(const DataObject *returnObject) const { return m_dataCopies.at(returnObject); }
	std::unordered_map<const DataObject *, const DataObject *> GetDataCopies() const { return m_dataCopies; }

	// Copy collection

	bool VisitIn(const AssignStatement *assignS) override;
	bool VisitIn(const CallExpression *call) override;

	// Debug printing

	std::string DebugString(unsigned int indent = 0);

private:
	const DataObjectAnalysis& m_objectAnalysis;
	const DataObject *GetDataObject(const CallExpression *call) const;

	std::unordered_map<const DataObject *, Initialization> m_dataInit;
	std::unordered_map<const DataObject *, const DataObject *> m_dataCopies;
};

}
}
