#pragma once

#include <utility>

#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"

#include "HorseIR/Analysis/DataObject/DataObjectAnalysis.h"
#include "HorseIR/Tree/Tree.h"

#include "Libraries/robin_hood.h"

namespace HorseIR {
namespace Analysis {

class DataInitializationAnalysis : public ConstHierarchicalVisitor
{
public:
	inline const static std::string Name = "Data initialization analysis";
	inline const static std::string ShortName = "datainit";

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
	robin_hood::unordered_map<const DataObject *, Initialization> GetDataInitializations() const { return m_dataInit; }

	bool ContainsDataCopy(const DataObject *returnObject) const
	{
		return (m_dataCopies.find(returnObject) != m_dataCopies.end());
	}
	const DataObject *GetDataCopy(const DataObject *returnObject) const { return m_dataCopies.at(returnObject); }
	robin_hood::unordered_map<const DataObject *, const DataObject *> GetDataCopies() const { return m_dataCopies; }

	// Copy collection

	bool VisitIn(const AssignStatement *assignS) override;
	bool VisitIn(const CallExpression *call) override;

	// Debug printing

	std::string DebugString(unsigned int indent = 0);

private:
	const DataObjectAnalysis& m_objectAnalysis;
	const DataObject *GetDataObject(const CallExpression *call) const;

	robin_hood::unordered_map<const DataObject *, Initialization> m_dataInit;
	robin_hood::unordered_map<const DataObject *, const DataObject *> m_dataCopies;
};

}
}
