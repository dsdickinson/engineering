# Grafana - Dashboards - Home Network - Node Exporters

I have written an article on Medium that explains the entire setup process for this. A few details from that article are included below.<br/>
https://medium.com/beyond-agile-leadership/grafana-monitor-your-home-network-3bfeae3f5885

## Purpose
I always want more visibility into my home network. It’s frustrating going in blind when issues arise and then you have to make un-educated guesses as to what the cause could be. When things start running slowly on my home network, I ask myself the following questions to start my Root Cause Analysis (RCA):

- What is it affecting?
- When did it start happening?
- What application or hardware is to blame?
- Are there logs that can be useful for troubleshooting?
- How can this be fully resolved?
- How do we prevent it from happening again?
<br/>

Running monitoring for devices and services are key to drilling down on these issues. This is where tools like Grafana paired with Prometheus shine. Setting up these monitoring tools can save a lot of time and energy.

In this tutorial, we will keep it pretty simple and just focus on setting up three devices to monitor. Adding or removing devices to monitoring is fairly trivial and I will give some examples of how to adjust those later. This article will also show the differences between setting up monitoring on a Mac device versus a Linux device versus a Windows device.

Note: I use a GitHub repo created by Francisco Galarza to facilitate the required software installs. It was very helpful and his accompanying YouTube video really is good viewing if you want to understand how Grafana & Prometheus work together. (See References below)

## Network Diagram
![image](https://github.com/user-attachments/assets/31db1eb0-8527-439d-9683-c08f690a1c55)


## Linux Dashboard
![image](https://github.com/user-attachments/assets/b9d95562-0741-4a7f-81fa-11fba6f00fe1)

## Mac OS Dashboard
![image](https://github.com/user-attachments/assets/631c54e8-3060-4a82-8e2f-47163fe9f87b)

## Windows Dashboard
![image](https://github.com/user-attachments/assets/3d0b88f4-82c3-42c1-aaf7-eb86fa7cce3b)
