import pandas as pd
import json
from collections import defaultdict

def check_climate_data():
    """Check climate dataset for potential mislabelings"""
    print("=" * 80)
    print("CHECKING CLIMATE DATASET")
    print("=" * 80)
    
    # Load all climate datasets
    datasets = {
        'train': 'data/climate_train.csv',
        'all': 'data/climate_all.csv',
        'test': 'data/climate_test.csv',
        'dev': 'data/climate_dev.csv'
    }
    
    all_issues = []
    
    for name, path in datasets.items():
        try:
            df = pd.read_csv(path)
            print(f"\n--- {name.upper()} Dataset ({len(df)} rows) ---")
            
            # Check for suspicious labels
            suspicious = []
            
            for idx, row in df.iterrows():
                # Check for NaN values
                if pd.isna(row['source_article']) or pd.isna(row['logical_fallacies']):
                    continue
                
                text = str(row['source_article']).lower()
                label = str(row['logical_fallacies']).strip()
                
                if not text or text == 'nan' or not label or label == 'nan':
                    continue
                
                issues = []
                
                # Check for factual reporting labeled as fallacies
                if label != 'intentional':
                    # Phrases that suggest factual reporting
                    factual_phrases = [
                        'according to', 'reported', 'study found', 'research shows',
                        'scientists say', 'experts say', 'analysis shows', 'data shows',
                        'modeling shows', 'forecast', 'project', 'estimate',
                        'economists project', 'computer forecasts', 'study from'
                    ]
                    
                    if any(phrase in text for phrase in factual_phrases):
                        issues.append("Contains factual reporting language but labeled as fallacy")
                
                # Check for ad populum mislabelings
                if label == 'ad populum':
                    # Ad populum should involve "many people believe" or similar
                    populum_indicators = [
                        'many people', 'everyone', 'most people', 'popular',
                        'common belief', 'widely believed', 'people believe',
                        'everyone else', 'majority'
                    ]
                    if not any(indicator in text for indicator in populum_indicators):
                        issues.append("Labeled ad populum but lacks popularity indicators")
                
                # Check for appeal to emotion mislabelings
                if label == 'appeal to emotion':
                    # Should have emotional language
                    emotion_indicators = [
                        'terrible', 'awful', 'devastating', 'catastrophic',
                        'fear', 'threat', 'danger', 'panic', 'crisis',
                        'emergency', 'chaos', 'mayhem', 'miserable',
                        'angry', 'scary', 'frightening'
                    ]
                    if not any(indicator in text for indicator in emotion_indicators):
                        # But could be subtle
                        pass
                
                # Check for false dilemma
                if label == 'false dilemma':
                    dilemma_indicators = ['either', 'or', 'only two', 'must choose', 'no alternative']
                    if not any(indicator in text for indicator in dilemma_indicators):
                        issues.append("Labeled false dilemma but lacks either/or structure")
                
                # Check for fallacy of credibility
                if label == 'fallacy of credibility':
                    # Should involve attacking credibility
                    credibility_indicators = [
                        'not credible', 'unreliable', 'biased', 'funded by',
                        'industry-funded', 'not trustworthy', 'questionable source',
                        'dispute', 'disputed', 'controversial'
                    ]
                    if not any(indicator in text for indicator in credibility_indicators):
                        issues.append("Labeled fallacy of credibility but may just be reporting")
                
                # Check for fallacy of relevance
                if label == 'fallacy of relevance':
                    # Hard to check automatically - would need context
                    pass
                
                # Check for faulty generalization
                if label == 'faulty generalization':
                    generalization_indicators = [
                        'all', 'every', 'always', 'never', 'none',
                        'must be', 'have to', 'will all', 'all of these'
                    ]
                    if not any(indicator in text for indicator in generalization_indicators):
                        # Could still be a generalization without these words
                        pass
                
                # Check for equivocation
                if label == 'equivocation':
                    # Hard to detect automatically - needs word meaning analysis
                    pass
                
                # Check for intentional
                if label == 'intentional':
                    # "Intentional" is tricky - it means intentionally using fallacious reasoning
                    # Hard to verify automatically
                    pass
                
                if issues:
                    suspicious.append({
                        'index': idx,
                        'text': row['source_article'][:200] + '...' if len(row['source_article']) > 200 else row['source_article'],
                        'label': label,
                        'issues': issues,
                        'url': row.get('original_url', 'N/A')
                    })
            
            print(f"Found {len(suspicious)} potentially mislabeled examples")
            
            if suspicious:
                print("\nPotentially Mislabeled Examples:")
                for item in suspicious[:20]:  # Show first 20
                    print(f"\n[{item['index']}] Label: {item['label']}")
                    print(f"    Text: {item['text']}")
                    print(f"    Issues: {', '.join(item['issues'])}")
                    print(f"    URL: {item['url']}")
                
                all_issues.extend(suspicious)
            
        except FileNotFoundError:
            print(f"\n--- {name.upper()} Dataset not found ---")
        except Exception as e:
            print(f"\n--- Error processing {name.upper()}: {e} ---")
    
    return all_issues

def check_edu_data():
    """Check education dataset for potential mislabelings"""
    print("\n" + "=" * 80)
    print("CHECKING EDUCATION DATASET")
    print("=" * 80)
    
    datasets = {
        'train': 'data/edu_train.csv',
        'test': 'data/edu_test.csv',
        'dev': 'data/edu_dev.csv',
        'all': 'data/edu_all.csv'
    }
    
    all_issues = []
    
    for name, path in datasets.items():
        try:
            df = pd.read_csv(path)
            print(f"\n--- {name.upper()} Dataset ({len(df)} rows) ---")
            
            # Check if source_article column exists
            if 'source_article' not in df.columns:
                print("  No source_article column found, skipping...")
                continue
            
            suspicious = []
            
            for idx, row in df.iterrows():
                text = str(row['source_article']).lower()
                label = str(row['updated_label']).strip() if 'updated_label' in df.columns else str(row.get('logical_fallacies', '')).strip()
                
                if pd.isna(text) or text == 'nan' or not text.strip():
                    continue
                
                issues = []
                
                # Similar checks as climate data
                if label == 'ad populum':
                    populum_indicators = [
                        'many people', 'everyone', 'most people', 'popular',
                        'common belief', 'widely believed', 'people believe',
                        'everyone else', 'majority', 'best-seller'
                    ]
                    if not any(indicator in text for indicator in populum_indicators):
                        issues.append("Labeled ad populum but lacks popularity indicators")
                
                if label == 'false causality':
                    causality_indicators = ['causes', 'caused', 'therefore', 'because', 'result of']
                    if not any(indicator in text for indicator in causality_indicators):
                        issues.append("Labeled false causality but lacks causal language")
                
                if label == 'false dilemma':
                    dilemma_indicators = ['either', 'or', 'must choose', 'only']
                    if not any(indicator in text for indicator in dilemma_indicators):
                        issues.append("Labeled false dilemma but lacks either/or structure")
                
                if label == 'circular reasoning':
                    circular_indicators = ['because', 'therefore', 'since']
                    # Hard to detect automatically
                    pass
                
                if issues:
                    suspicious.append({
                        'index': idx,
                        'text': text[:200] + '...' if len(text) > 200 else text,
                        'label': label,
                        'issues': issues
                    })
            
            print(f"Found {len(suspicious)} potentially mislabeled examples")
            
            if suspicious:
                print("\nPotentially Mislabeled Examples:")
                for item in suspicious[:20]:
                    print(f"\n[{item['index']}] Label: {item['label']}")
                    print(f"    Text: {item['text']}")
                    print(f"    Issues: {', '.join(item['issues'])}")
                
                all_issues.extend(suspicious)
                
        except FileNotFoundError:
            print(f"\n--- {name.upper()} Dataset not found ---")
        except Exception as e:
            print(f"\n--- Error processing {name.upper()}: {e} ---")
    
    return all_issues

def analyze_specific_problematic_cases():
    """Analyze specific cases that look problematic"""
    print("\n" + "=" * 80)
    print("ANALYZING SPECIFIC PROBLEMATIC CASES")
    print("=" * 80)
    
    problematic_cases = []
    
    # Load climate_all.csv
    try:
        df = pd.read_csv('data/climate_all.csv')
        
        # Case 1: Factual reporting labeled as fallacies
        print("\n1. Checking factual reporting labeled as fallacies:")
        factual_labels = ['fallacy of credibility', 'ad populum', 'appeal to emotion']
        
        for idx, row in df.iterrows():
            text = str(row['source_article']).lower()
            label = str(row['logical_fallacies']).strip()
            
            if label in factual_labels:
                # Check if it's just reporting facts
                if 'according to' in text or 'study found' in text or 'research shows' in text:
                    if 'modeling shows' in text or 'project' in text or 'forecast' in text:
                        problematic_cases.append({
                            'type': 'Factual reporting labeled as fallacy',
                            'text': row['source_article'][:150],
                            'label': label,
                            'row': idx
                        })
        
        # Case 2: Ad populum without popularity indicators
        print("\n2. Checking ad populum without popularity indicators:")
        ad_populum_rows = df[df['logical_fallacies'] == 'ad populum']
        
        for idx, row in ad_populum_rows.iterrows():
            text = str(row['source_article']).lower()
            populum_words = ['many people', 'everyone', 'most people', 'popular', 'majority', 'people believe']
            
            if not any(word in text for word in populum_words):
                problematic_cases.append({
                    'type': 'Ad populum without popularity indicators',
                    'text': row['source_article'][:150],
                    'label': 'ad populum',
                    'row': idx
                })
        
        print(f"\nFound {len(problematic_cases)} problematic cases")
        
        for i, case in enumerate(problematic_cases[:15], 1):
            print(f"\n[{i}] Type: {case['type']}")
            print(f"    Label: {case['label']}")
            print(f"    Text: {case['text']}")
            print(f"    Row: {case['row']}")
            
    except Exception as e:
        print(f"Error: {e}")
    
    return problematic_cases

def generate_report():
    """Generate a comprehensive report"""
    print("\n" + "=" * 80)
    print("DATA VALIDATION REPORT")
    print("=" * 80)
    
    climate_issues = check_climate_data()
    edu_issues = check_edu_data()
    problematic = analyze_specific_problematic_cases()
    
    # Save report
    report = {
        'climate_issues': len(climate_issues),
        'education_issues': len(edu_issues),
        'problematic_cases': len(problematic),
        'total_issues': len(climate_issues) + len(edu_issues) + len(problematic),
        'details': {
            'climate_mislabelings': climate_issues[:50],  # First 50
            'education_mislabelings': edu_issues[:50],
            'problematic_cases': problematic[:50]
        }
    }
    
    with open('label_validation_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Climate dataset issues: {len(climate_issues)}")
    print(f"Education dataset issues: {len(edu_issues)}")
    print(f"Problematic cases: {len(problematic)}")
    print(f"Total issues found: {report['total_issues']}")
    print(f"\nFull report saved to: label_validation_report.json")

if __name__ == "__main__":
    generate_report()

