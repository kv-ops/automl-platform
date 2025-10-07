# 🛡️ Production Universal ML Agent - Usage Guide

## 📋 Table des Matières

1. [Vue d'ensemble](#vue-densemble)
2. [Protections Mémoire](#protections-mémoire)
3. [Exemples d'Utilisation](#exemples-dutilisation)
4. [Configuration Production](#configuration-production)
5. [Monitoring & Alertes](#monitoring--alertes)
6. [Troubleshooting](#troubleshooting)

---

## Vue d'ensemble

Le **Production Universal ML Agent** est une version enterprise-grade avec :

### ✨ Nouvelles Fonctionnalités

| Feature | Description | Impact |
|---------|-------------|--------|
| **Memory Monitoring** | Surveillance en temps réel | Évite les OOM crashes |
| **LRU Cache** | Cache intelligent avec limites | Réduit les appels répétés |
| **Batch Processing** | Traitement par lots | Gère les gros datasets |
| **Memory Budgets** | Limites par opération | Contrôle granulaire |
| **Auto Cleanup** | Garbage collection automatique | Libère la mémoire proactivement |

### 🎯 Cas d'Usage

- **Production ML pipelines** nécessitant haute disponibilité
- **Large datasets** (> 1M rows) nécessitant batch processing
- **Environnements contraints** (Cloud, containers avec memory limits)
- **Long-running services** nécessitant stability

---

## Protections Mémoire

### 1. Memory Monitor

Surveillance continue avec seuils configurables :

```python
from automl_platform.agents import ProductionUniversalMLAgent

# Configuration des seuils
agent = ProductionUniversalMLAgent(
    memory_warning_mb=1000,   # Warning à 1GB
    memory_critical_mb=2000,  # Critical à 2GB
    max_cache_mb=500          # Cache limité à 500MB
)

# Monitoring automatique à chaque étape
status = agent.memory_monitor.check_memory()
print(f"Memory: {status['current_mb']:.1f} MB")
print(f"Peak: {status['peak_mb']:.1f} MB")
print(f"Available: {status['available_mb']:.1f} MB")
```

**Comportement :**
- ⚠️ **WARNING** (1GB) : Log d'alerte, continue
- 🔴 **CRITICAL** (2GB) : Log critique, force cleanup
- 💥 **OOM** : Exception Python, graceful degradation

### 2. LRU Cache avec Limites

Cache intelligent qui évicte automatiquement les anciennes entrées :

```python
# Le cache est automatique, mais vous pouvez le contrôler
agent.cache.clear()  # Vider complètement

# Statistiques
stats = agent.cache.get_stats()
print(f"Cache: {stats['items']} items")
print(f"Size: {stats['size_mb']:.1f} MB")
print(f"Utilization: {stats['utilization']:.1%}")
```

**Ce qui est caché :**
- Contextes ML détectés (par hash de données)
- Best practices (par problem_type)
- Profils de données
- Résultats de validation

### 3. Memory Budgets

Chaque opération a un budget mémoire :

```python
from automl_platform.agents.universal_ml_agent_production import MemoryBudget

# Utilisation manuelle pour vos fonctions
async def my_operation(df):
    with MemoryBudget(budget_mb=500):
        # Votre code ici
        result = expensive_operation(df)
    return result
```

**Opérations protégées :**
- `understand_problem()` : 500 MB
- `validate_with_standards()` : 300 MB
- `generate_optimal_config()` : 400 MB
- `execute_intelligent_cleaning()` : 1000 MB

### 4. Batch Processing

Pour datasets > 20K rows :

```python
agent = ProductionUniversalMLAgent(
    batch_size=10000  # Traiter par lots de 10K
)

# Automatique si df > 20K rows
result = await agent.automl_without_templates(
    df=large_df,  # 1M rows
    target_col='target'
)
# ✅ Traité en 100 batches automatiquement
```

---

## Exemples d'Utilisation

### Exemple 1 : Pipeline Standard

```python
import pandas as pd
from automl_platform.agents import ProductionUniversalMLAgent

# Initialiser
agent = ProductionUniversalMLAgent(
    use_claude=True,
    max_cache_mb=500,
    memory_warning_mb=1000,
    memory_critical_mb=2000
)

# Charger données
df = pd.read_csv('customer_churn.csv')

# Exécuter pipeline
result = await agent.automl_without_templates(
    df=df,
    target_col='churn',
    user_hints={'problem_type': 'churn_prediction'},
    constraints={'time_budget': 3600}
)

# Résultats avec métriques mémoire
print(f"Success: {result.success}")
print(f"Execution time: {result.execution_time:.1f}s")
print(f"Memory delta: {result.memory_stats['delta_mb']:.1f} MB")
print(f"Peak memory: {result.memory_stats['peak_mb']:.1f} MB")
print(f"Cache hit rate: {result.performance_profile['cache_hit_rate']:.1%}")

if result.claude_summary:
    print(f"\n💎 Claude Summary:\n{result.claude_summary}")
```

### Exemple 2 : Large Dataset (1M+ rows)

```python
# Configuration pour gros dataset
agent = ProductionUniversalMLAgent(
    use_claude=True,
    max_cache_mb=1000,        # Plus de cache
    memory_warning_mb=2000,   # Seuils plus hauts
    memory_critical_mb=4000,
    batch_size=5000           # Petits batches
)

# Charger en chunks pour économiser mémoire
chunks = []
for chunk in pd.read_csv('huge_dataset.csv', chunksize=50000):
    chunks.append(chunk)

df = pd.concat(chunks, ignore_index=True)
del chunks  # Libérer mémoire

# Pipeline avec batch processing automatique
result = await agent.automl_without_templates(
    df=df,
    target_col='target'
)

# Résultat avec stats détaillées
print(f"Rows processed: {len(result.cleaned_data)}")
print(f"Batches used: {result.cleaning_report.get('n_batches', 'N/A')}")
print(f"Memory efficient: {result.memory_stats['delta_mb'] < 1000}")
```

### Exemple 3 : Environnement Contraint (Docker)

```python
# Configuration minimale pour container avec 2GB RAM
agent = ProductionUniversalMLAgent(
    use_claude=False,          # Économiser mémoire
    max_cache_mb=100,          # Cache minimal
    memory_warning_mb=500,     # Seuils bas
    memory_critical_mb=1500,
    batch_size=2000            # Très petits batches
)

# Forcer cleanup régulier
import gc

result = await agent.automl_without_templates(df, target_col='y')

# Cleanup manuel après utilisation
agent.cleanup()
gc.collect()
```

### Exemple 4 : Long-Running Service

```python
import asyncio

agent = ProductionUniversalMLAgent(
    use_claude=True,
    max_cache_mb=500
)

async def process_datasets():
    datasets = [
        ('customers.csv', 'churn'),
        ('transactions.csv', 'fraud'),
        ('products.csv', 'category')
    ]
    
    for file, target in datasets:
        df = pd.read_csv(file)
        
        result = await agent.automl_without_templates(
            df=df,
            target_col=target
        )
        
        print(f"✅ {file}: {result.success}")
        
        # Cleanup entre chaque dataset
        if agent.memory_monitor.get_memory_usage_mb() > 1000:
            agent.cache.clear()
            gc.collect()
            print("🧹 Cache cleared")
        
        # Petit délai
        await asyncio.sleep(1)

# Exécuter
await process_datasets()

# Cleanup final
agent.cleanup()
```

---

## Configuration Production

### Variables d'Environnement

```bash
# .env
CLAUDE_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Memory settings
AUTOML_MAX_CACHE_MB=500
AUTOML_MEMORY_WARNING_MB=1000
AUTOML_MEMORY_CRITICAL_MB=2000
AUTOML_BATCH_SIZE=10000

# Performance
AUTOML_MAX_WORKERS=4
AUTOML_ENABLE_PROFILING=true
```

### Docker Configuration

```dockerfile
FROM python:3.10-slim

# Limiter mémoire container
ENV PYTHONUNBUFFERED=1
ENV AUTOML_MAX_CACHE_MB=200
ENV AUTOML_MEMORY_WARNING_MB=500
ENV AUTOML_MEMORY_CRITICAL_MB=1500

WORKDIR /app

# Installer dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier code
COPY . .

# Limiter mémoire Python
CMD ["python", "-X", "dev", "-W", "ignore", "main.py"]
```

```yaml
# docker-compose.yml
services:
  automl:
    build: .
    mem_limit: 2g
    mem_reservation: 1g
    environment:
      - AUTOML_MAX_CACHE_MB=200
      - AUTOML_MEMORY_CRITICAL_MB=1500
```

### Paramétrage du mode hybride retail

Activez l'arbitrage local/agents directement depuis `config.yaml` pour les jeux de données retail :

```yaml
agent_first:
  enable_hybrid_mode: true
  hybrid_mode_thresholds:
    missing_threshold: 0.35
    quality_score_threshold: 70.0
  retail_rules:
    gs1_compliance_target: 0.98
  hybrid_cost_limits:
    max_total: 5.0
```

> 💡 **Bonnes pratiques QA** : lancer `pytest tests/test_agents.py -k should_use_agent_for_retail_risks` et `pytest tests/test_data_quality_agent.py -k retail_specific` avant toute mise en production pour vérifier les règles retail et le rapport final.

### Kubernetes Configuration

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: automl-agent
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: automl
        image: automl-agent:latest
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        env:
        - name: AUTOML_MAX_CACHE_MB
          value: "300"
        - name: AUTOML_MEMORY_CRITICAL_MB
          value: "1800"
```

---

## Monitoring & Alertes

### Métriques à Surveiller

```python
# Extraire métriques pour monitoring externe
metrics = {
    'memory': {
        'current_mb': result.memory_stats['final_mb'],
        'peak_mb': result.memory_stats['peak_mb'],
        'delta_mb': result.memory_stats['delta_mb']
    },
    'cache': {
        'items': result.cache_stats['items'],
        'size_mb': result.cache_stats['size_mb'],
        'hit_rate': result.performance_profile['cache_hit_rate']
    },
    'performance': {
        'execution_time': result.execution_time,
        'success': result.success,
        'agent_calls': result.performance_profile['agent_calls']
    }
}

# Envoyer à Prometheus, DataDog, etc.
send_to_monitoring(metrics)
```

### Logs Structurés

Le code log déjà en format structuré :

```python
import logging
import json

# Configure structured logging
class StructuredLogger(logging.Handler):
    def emit(self, record):
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'memory_mb': agent.memory_monitor.get_memory_usage_mb()
        }
        print(json.dumps(log_entry))

handler = StructuredLogger()
logging.getLogger().addHandler(handler)
```

### Alertes Recommandées

| Alert | Condition | Action |
|-------|-----------|--------|
| High Memory | > 80% limit | Scale up or clear cache |
| Low Cache Hit | < 30% | Review caching strategy |
| Long Execution | > 10min | Check batch size |
| Frequent OOM | > 1/day | Increase memory limits |

---

## Troubleshooting

### Problème : OOM (Out of Memory)

**Symptômes :**
- Process killed par OS
- `MemoryError` exceptions

**Solutions :**
```python
# 1. Réduire batch size
agent = ProductionUniversalMLAgent(batch_size=1000)

# 2. Réduire cache
agent = ProductionUniversalMLAgent(max_cache_mb=100)

# 3. Forcer cleanup régulier
agent.cleanup()
gc.collect()

# 4. Désactiver Claude si nécessaire
agent = ProductionUniversalMLAgent(use_claude=False)
```

### Problème : Slow Performance

**Symptômes :**
- Exécution > 10 minutes
- Low cache hit rate

**Solutions :**
```python
# 1. Augmenter cache
agent = ProductionUniversalMLAgent(max_cache_mb=1000)

# 2. Réutiliser l'agent entre calls
agent = ProductionUniversalMLAgent()
for df in datasets:
    result = await agent.automl_without_templates(df)
    # agent garde le cache chaud

# 3. Vérifier batch size
agent = ProductionUniversalMLAgent(batch_size=20000)  # Plus gros
```

### Problème : Memory Leaks

**Symptômes :**
- Mémoire croissante sur multiple runs
- Never releases memory

**Solutions :**
```python
# 1. Cleanup explicite
agent.cleanup()

# 2. Utiliser context manager
async with create_agent() as agent:
    result = await agent.automl_without_templates(df)
# Cleanup automatique

# 3. Vérifier références circulaires
import gc
gc.set_debug(gc.DEBUG_LEAK)
```

### Problème : Cache Thrashing

**Symptômes :**
- Cache hit rate < 20%
- Frequent evictions

**Solutions :**
```python
# 1. Augmenter taille cache
agent = ProductionUniversalMLAgent(max_cache_mb=1000)

# 2. Organiser les datasets similaires ensemble
similar_datasets = [df1, df2, df3]  # Même structure
for df in similar_datasets:
    result = await agent.automl_without_templates(df)

# 3. Pre-warm cache
await agent.understand_problem(df_template)
```

---

## Best Practices

### ✅ DO

1. **Monitoring actif** : Surveiller mémoire en production
2. **Cleanup régulier** : `agent.cleanup()` entre batches
3. **Batch approprié** : Adapter `batch_size` aux données
4. **Cache sizing** : Dimensionner selon workload
5. **Logs structurés** : Pour debugging efficace

### ❌ DON'T

1. **Ignorer warnings** : Toujours investiguer les warnings mémoire
2. **Cache illimité** : Toujours définir `max_cache_mb`
3. **Batches géants** : Ne pas mettre `batch_size > 50000`
4. **Ignorer cleanup** : Ne jamais oublier le cleanup
5. **Réinstancier l'agent** : Réutiliser pour cache warm

---

## Performance Benchmarks

### Small Dataset (< 10K rows)

```
Memory: ~200 MB
Time: 30-60s
Cache Hit: 60-80%
Batch: N/A (single pass)
```

### Medium Dataset (10K-100K rows)

```
Memory: ~500 MB
Time: 2-5 min
Cache Hit: 40-60%
Batch: 2-5 batches
```

### Large Dataset (100K-1M rows)

```
Memory: ~1 GB
Time: 10-20 min
Cache Hit: 20-40%
Batch: 10-100 batches
```

### Very Large Dataset (> 1M rows)

```
Memory: ~2 GB
Time: 30-60 min
Cache Hit: 10-30%
Batch: 100+ batches
```

---

## Support & Resources

- **Documentation**: `/docs/production_guide.md`
- **Examples**: `/examples/production/`
- **Issues**: GitHub Issues
- **Monitoring**: Grafana dashboards in `/monitoring/`

---

**Version**: 1.0.0-production  
**Last Updated**: October 2025  
**License**: MIT