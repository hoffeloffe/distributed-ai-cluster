{{/*
Expand the name of the chart.
*/}}
{{- define "distributed-ai-cluster.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "distributed-ai-cluster.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "distributed-ai-cluster.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "distributed-ai-cluster.labels" -}}
helm.sh/chart: {{ include "distributed-ai-cluster.chart" . }}
{{ include "distributed-ai-cluster.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "distributed-ai-cluster.selectorLabels" -}}
app.kubernetes.io/name: {{ include "distributed-ai-cluster.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "distributed-ai-cluster.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "distributed-ai-cluster.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Generate full image name with registry
*/}}
{{- define "distributed-ai-cluster.master.image" -}}
{{- if .Values.cluster.master.image.registry }}
{{- printf "%s/%s:%s" .Values.cluster.master.image.registry .Values.cluster.master.image.repository .Values.cluster.master.image.tag }}
{{- else }}
{{- printf "%s:%s" .Values.cluster.master.image.repository .Values.cluster.master.image.tag }}
{{- end }}
{{- end }}

{{- define "distributed-ai-cluster.worker.image" -}}
{{- if .Values.cluster.workers.image.registry }}
{{- printf "%s/%s:%s" .Values.cluster.workers.image.registry .Values.cluster.workers.image.repository .Values.cluster.workers.image.tag }}
{{- else }}
{{- printf "%s:%s" .Values.cluster.workers.image.repository .Values.cluster.workers.image.tag }}
{{- end }}
{{- end }}

{{/*
Get the namespace
*/}}
{{- define "distributed-ai-cluster.namespace" -}}
{{- default .Release.Namespace .Values.namespace }}
{{- end }}
