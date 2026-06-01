import pandas as pd

class CourseRecommender:
    def __init__(self, rules: pd.DataFrame):
        self.rules = rules.copy()
        
        self.rules["full_itemset"] = self.rules.apply(
            lambda row: frozenset(row["antecedents"].union(row["consequents"])), axis=1
        )
        self.rules["base_itemset"] = self.rules["full_itemset"].apply(
            lambda x: frozenset([item.split('_')[0] for item in x])
        )
        self.rules["full_size"] = self.rules["base_itemset"].apply(len)
        
        self.rules["is_all_approved"] = self.rules["full_itemset"].apply(
            lambda x: all("_Aprovado" in item for item in x)
        )
        self.rules["has_reprovado"] = self.rules["full_itemset"].apply(
            lambda x: any("_Reprovado" in item for item in x)
        )

    def simulate_scenario(self, target_base_courses: set, target_size: int):
        mask = self.rules["base_itemset"].apply(lambda x: target_base_courses.issubset(x))
        relevant_rules = self.rules[mask]

        if relevant_rules.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None, None

        grouped = relevant_rules.groupby("base_itemset")
        scenario_stats = []
        
        for base_itemset, group in grouped:
            success_rules = group[group["is_all_approved"]]
            risk_rules = group[group["has_reprovado"]]
            
            s_support = success_rules["support"].max() if not success_rules.empty else 0
            s_conf = success_rules["confidence"].max() if not success_rules.empty else 0
            
            r_support = risk_rules["support"].max() if not risk_rules.empty else 0
            r_conf = risk_rules["confidence"].max() if not risk_rules.empty else 0
            
            s_itemset = success_rules.sort_values("support", ascending=False).iloc[0]["full_itemset"] if not success_rules.empty else set()
            r_itemset = risk_rules.sort_values("support", ascending=False).iloc[0]["full_itemset"] if not risk_rules.empty else set()

            scenario_stats.append({
                "base_itemset": base_itemset,
                "size": len(base_itemset),
                "success_support": s_support,
                "success_conf": s_conf,
                "risk_support": r_support,
                "risk_conf": r_conf,
                "success_itemset": s_itemset,
                "risk_itemset": r_itemset
            })

        df_stats = pd.DataFrame(scenario_stats)
        if df_stats.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None, None

        df_stats["max_overall_support"] = df_stats[["success_support", "risk_support"]].max(axis=1)
        
        analise_geral = df_stats.sort_values("max_overall_support", ascending=False).drop_duplicates("size")
        analise_aprovacao = df_stats[df_stats["success_support"] > 0].sort_values("success_support", ascending=False).drop_duplicates("size")
        analise_reprovacao = df_stats[df_stats["risk_support"] > 0].sort_values("risk_support", ascending=False).drop_duplicates("size")

        df_target = df_stats[df_stats["size"] == target_size].copy()
        best_scenario = None
        worst_scenario = None

        if not df_target.empty:
            best_candidates = df_target[df_target["success_support"] > 0].sort_values(
                by=["success_support", "success_conf"], ascending=[False, False]
            )
            if not best_candidates.empty:
                best_scenario = best_candidates.iloc[0].to_dict()

            risk_candidates = df_target[df_target["risk_support"] > 0].sort_values(
                by=["risk_support", "risk_conf"], ascending=[False, False]
            )
            
            if best_scenario:
                risk_candidates = risk_candidates[risk_candidates["base_itemset"] != best_scenario["base_itemset"]]

            if not risk_candidates.empty:
                worst_scenario = risk_candidates.iloc[0].to_dict()

        return analise_geral, analise_aprovacao, analise_reprovacao, best_scenario, worst_scenario